"""
RAG That Reads, Sees, and Hears — core pipeline.

Ingests text, URLs, PDFs, images, audio, and video into ChromaDB. Retrieves
with Gemini Embedding 2 and answers with Gemini 3 Flash, passing the original
file back to the model for image, audio, and video sources so it sees the
media rather than only a description of it.
"""

from __future__ import annotations

import mimetypes
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import httpx
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ── Constants ──────────────────────────────────────────────────────────────────

EMBED_MODEL = "gemini-embedding-2"
GEN_MODEL = "gemini-3-flash-preview"
COLLECTION_PREFIX = "multimodal_rag"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

MAX_PDF_PAGES = 100        # cap so a 900-page book doesn't stall ingestion
EMBED_WORKERS = 4          # parallel embedding calls; raise if you have headroom
UPLOAD_TIMEOUT = 300       # seconds to wait for the File API to finish processing

MEDIA_PROMPTS = {
    "image": (
        "Describe this image in full detail. Include all visible text, objects, "
        "people, scenes, colors, spatial layout, and any other relevant information."
    ),
    "audio": (
        "Transcribe and describe this audio in full. Include all speech (verbatim), "
        "background sounds, music, tone, and any other auditory elements."
    ),
    "video": (
        "Describe and summarize this video in detail. Cover visual scenes, any "
        "on-screen text, speech (verbatim where possible), audio, and key events "
        "with approximate timestamps."
    ),
}

MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".m4a": "audio/mp4", ".flac": "audio/flac", ".aac": "audio/aac",
    ".mp4": "video/mp4", ".avi": "video/x-msvideo",
    ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
}

# Short tags used by the UI in place of icons.
SOURCE_TAGS = {
    "text": "TXT", "url": "URL", "pdf": "PDF",
    "image": "IMG", "audio": "AUD", "video": "VID",
}


# ── LangChain-compatible Gemini embeddings ─────────────────────────────────────

class GeminiEmbeddings(Embeddings):
    """Wraps Gemini Embedding 2 for use with LangChain components."""

    def __init__(self, client: genai.Client):
        self.client = client

    def _embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(model=EMBED_MODEL, contents=text)
        return list(result.embeddings[0].values)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        One request per chunk, run on a small thread pool.

        The endpoint takes a single input at a time, so a 100-page PDF is
        several hundred calls. Threading keeps that from being several hundred
        round trips end to end. ThreadPoolExecutor.map preserves input order,
        which matters because the vectors are zipped back onto the documents.
        """
        if len(texts) <= 1:
            return [self._embed(t) for t in texts]
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
            return list(pool.map(self._embed, texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class IndexedSource:
    label: str
    source_type: str  # text | url | pdf | image | audio | video
    chunks: int
    file_uri: str = ""
    mime_type: str = ""


@dataclass
class RAGResult:
    answer: str
    question: str
    retrieved_docs: list[Document] = field(default_factory=list)


# ── Main class ─────────────────────────────────────────────────────────────────

class MultimodalRAG:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.embeddings = GeminiEmbeddings(self.client)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        # In-memory collection, unique per instance. chromadb.Client() hands back
        # a shared in-process system, so a fixed collection name would collide
        # the second time this class is constructed (i.e. on "Start over").
        self._chroma = chromadb.Client()
        self._collection = self._chroma.create_collection(
            name=f"{COLLECTION_PREFIX}_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},
        )

        self.sources: list[IndexedSource] = []

    # ── Private helpers ────────────────────────────────────────────────────────

    def _store_docs(self, docs: list[Document]) -> int:
        """Embed and insert LangChain Documents into ChromaDB."""
        if not docs:
            return 0
        texts = [d.page_content for d in docs]
        vectors = self.embeddings.embed_documents(texts)
        self._collection.add(
            ids=[str(uuid.uuid4()) for _ in docs],
            embeddings=vectors,
            documents=texts,
            metadatas=[d.metadata for d in docs],
        )
        return len(docs)

    def _upload_file(self, file_path: str, mime_type: str) -> str:
        """Upload to the Gemini File API, wait for processing, return the URI."""
        ref = self.client.files.upload(
            file=file_path,
            config=types.UploadFileConfig(mime_type=mime_type),
        )
        deadline = time.time() + UPLOAD_TIMEOUT
        while ref.state.name == "PROCESSING":
            if time.time() > deadline:
                raise TimeoutError(
                    f"Gemini is still processing this file after {UPLOAD_TIMEOUT}s. "
                    "Try a shorter or smaller file."
                )
            time.sleep(2)
            ref = self.client.files.get(name=ref.name)

        if ref.state.name == "FAILED":
            raise RuntimeError("Gemini could not process this file.")
        return ref.uri

    def _describe_media(self, file_uri: str, mime_type: str, media_type: str) -> str:
        """Ask Gemini to describe or transcribe a media file via its URI."""
        prompt = MEDIA_PROMPTS.get(media_type, "Describe this content in detail.")
        response = self.client.models.generate_content(
            model=GEN_MODEL,
            contents=[
                types.Part(file_data=types.FileData(file_uri=file_uri, mime_type=mime_type)),
                types.Part(text=prompt),
            ],
        )
        return response.text.strip()

    def _get_mime(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        return MIME_MAP.get(ext, mimetypes.guess_type(file_path)[0] or "application/octet-stream")

    def _add_media(self, file_path: str, media_type: str) -> int:
        """
        Shared path for image, audio, and video.

        Upload the file, have Gemini write a description, index that text, and
        keep the file URI on every chunk so the real media can be handed back
        to the model at query time.
        """
        label = Path(file_path).name
        mime_type = self._get_mime(file_path)
        file_uri = self._upload_file(file_path, mime_type)
        description = self._describe_media(file_uri, mime_type, media_type)

        docs = self.splitter.create_documents(
            [description],
            metadatas=[{
                "source_type": media_type, "source_label": label,
                "file_uri": file_uri, "mime_type": mime_type,
            }],
        )
        n = self._store_docs(docs)
        self.sources.append(IndexedSource(
            label=label, source_type=media_type, chunks=n,
            file_uri=file_uri, mime_type=mime_type,
        ))
        return n

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_text(self, text: str, label: str = "Pasted text") -> int:
        """Index raw text."""
        docs = self.splitter.create_documents(
            [text],
            metadatas=[{"source_type": "text", "source_label": label}],
        )
        n = self._store_docs(docs)
        self.sources.append(IndexedSource(label=label, source_type="text", chunks=n))
        return n

    def add_url(self, url: str) -> int:
        """Fetch a page, strip the HTML furniture, index what's left."""
        r = httpx.get(url, timeout=20, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

        docs = self.splitter.create_documents(
            [text],
            metadatas=[{"source_type": "url", "source_label": url}],
        )
        n = self._store_docs(docs)
        self.sources.append(IndexedSource(label=url, source_type="url", chunks=n))
        return n

    def add_pdf(self, file_path: str) -> int:
        """Load a PDF page by page and index the first MAX_PDF_PAGES of it."""
        label = Path(file_path).name
        pages = PyPDFLoader(file_path).load()
        if len(pages) > MAX_PDF_PAGES:
            pages = pages[:MAX_PDF_PAGES]

        chunks = self.splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata.update({"source_type": "pdf", "source_label": label})
        n = self._store_docs(chunks)
        self.sources.append(IndexedSource(label=label, source_type="pdf", chunks=n))
        return n

    def add_image(self, file_path: str) -> int:
        return self._add_media(file_path, "image")

    def add_audio(self, file_path: str) -> int:
        return self._add_media(file_path, "audio")

    def add_video(self, file_path: str) -> int:
        return self._add_media(file_path, "video")

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """
        Retrieve the closest chunks and generate a grounded answer.

        When a retrieved chunk came from an image, audio, or video source, the
        original file is attached to the request as well, so Gemini works from
        the media itself and not just the description written at ingest time.
        """
        if self._collection.count() == 0:
            return RAGResult(
                answer="Nothing has been indexed yet. Add a source from the left "
                       "to start asking questions.",
                question=question,
            )

        query_vec = self.embeddings.embed_query(question)
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas"],
        )
        retrieved = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

        parts: list[types.Part] = [
            types.Part(text=(
                "You are a knowledgeable assistant. Answer the question using ONLY "
                "the provided context. Cite the source label for each piece of "
                "information you use. If the answer is not in the context, say so.\n"
            ))
        ]

        seen_uris: set[str] = set()
        context_texts: list[str] = []

        for doc in retrieved:
            meta = doc.metadata
            label = meta.get("source_label", "unknown")
            file_uri = meta.get("file_uri", "")
            mime_type = meta.get("mime_type", "")

            # Attach each distinct media file once, however many of its chunks hit.
            if file_uri and file_uri not in seen_uris:
                parts.append(
                    types.Part(file_data=types.FileData(file_uri=file_uri, mime_type=mime_type))
                )
                seen_uris.add(file_uri)

            context_texts.append(f"[Source: {label}]\n{doc.page_content}")

        parts.append(types.Part(text="Context:\n\n" + "\n\n---\n\n".join(context_texts)))
        parts.append(types.Part(text=f"\nQuestion: {question}\n\nAnswer (with citations):"))

        response = self.client.models.generate_content(model=GEN_MODEL, contents=parts)

        return RAGResult(
            answer=response.text.strip(),
            question=question,
            retrieved_docs=retrieved,
        )

    # ── Stats ──────────────────────────────────────────────────────────────────

    def chunk_count(self) -> int:
        return self._collection.count()

    def source_count(self) -> int:
        return len(self.sources)

    def media_count(self) -> int:
        """How many indexed sources carry a real file the model can re-open."""
        return sum(1 for s in self.sources if s.file_uri)
