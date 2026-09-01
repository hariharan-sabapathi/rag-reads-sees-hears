"""
RAG That Reads, Sees, and Hears — Streamlit UI.
"""

import contextlib
import html
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag import SOURCE_TAGS, MultimodalRAG

load_dotenv()

st.set_page_config(
    page_title="RAG That Reads, Sees, and Hears",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Look and feel ──────────────────────────────────────────────────────────────

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&family=Martian+Mono:wght@400;500&display=swap');

:root {
  --page:#FFFFFF; --panel:#F7F5F1; --raise:#F2EFE9; --line:#E3DED5;
  --text:#221E19; --dim:#645C53; --faint:#958C80;
  --signal:#A26F16; --live:#2E6E5B; --alert:#B14330;
}

html, body, [class*="css"], .stApp { font-family:'Archivo', system-ui, sans-serif; }
.stApp { background:var(--page); color:var(--text); }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:2.4rem; max-width:960px; }

[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--text); }
[data-testid="stSidebar"] hr { border-color:var(--line); margin:14px 0; }
[data-testid="stSidebar"] label { color:var(--dim) !important; font-size:13px !important; }

/* Masthead */
.mast { margin-bottom:6px; }
.mast h1 {
  font-family:'Archivo'; font-size:34px; font-weight:600; letter-spacing:-0.02em;
  margin:0 0 6px; color:var(--text);
}
.mast p { color:var(--dim); font-size:15px; line-height:1.5; margin:0; max-width:62ch; }

/* Counter strip */
.strip { display:flex; gap:28px; padding:16px 0 18px; border-bottom:1px solid var(--line); margin-bottom:26px; }
.stat .n { font-family:'Martian Mono'; font-size:19px; color:var(--signal); font-weight:500; }
.stat .k { font-size:12px; color:var(--faint); margin-top:3px; }

/* Register rows */
.reg-head { font-size:12px; color:var(--faint); margin:2px 0 9px; }
.row {
  display:flex; align-items:center; gap:9px; padding:8px 10px; margin-bottom:5px;
  background:var(--raise); border:1px solid var(--line); border-radius:3px;
}
.tag {
  font-family:'Martian Mono'; font-size:9.5px; letter-spacing:.04em; padding:3px 5px;
  border:1px solid var(--faint); color:var(--dim); border-radius:2px; flex:none;
}
.row.media .tag { border-color:var(--live); color:var(--live); }
.row .lab { flex:1; font-size:12.5px; color:var(--text); overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; }
.row .cnt { font-family:'Martian Mono'; font-size:10px; color:var(--faint); flex:none; }
.legend { font-size:11.5px; color:var(--faint); margin-top:9px; line-height:1.5; }
.legend b { color:var(--live); font-weight:500; }

/* Status */
.state { font-size:12.5px; display:flex; align-items:center; gap:7px; }
.state i { width:6px; height:6px; border-radius:50%; background:var(--live); font-style:normal; }
.state.off i { background:var(--alert); }
.state.off { color:var(--dim); }

/* Chat */
[data-testid="stChatMessage"] { background:transparent !important; border:0 !important; padding:0 !important;
  margin-bottom:20px !important; }
[data-testid="stChatMessageAvatar"], [data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] { display:none !important; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color:var(--text); line-height:1.65; }
[data-testid="stChatMessage"] code { background:var(--raise) !important; color:var(--signal) !important; }
.turn-q { font-size:17px; font-weight:500; color:var(--text); padding-left:13px;
  border-left:2px solid var(--signal); margin-bottom:18px; }

[data-testid="stChatInput"] textarea {
  background:var(--panel) !important; border:1px solid var(--line) !important;
  color:var(--text) !important; border-radius:3px !important; font-family:'Archivo' !important; }
[data-testid="stChatInput"] textarea:focus { border-color:var(--signal) !important; box-shadow:none !important; }
[data-testid="stChatInput"] textarea::placeholder { color:var(--faint) !important; }

/* Retrieved chunks */
.chunk { background:var(--panel); border:1px solid var(--line); border-left:2px solid var(--signal);
  padding:11px 14px; margin-bottom:7px; font-size:13px; color:var(--dim); line-height:1.55; }
.chunk.media { border-left-color:var(--live); }
.chunk .from { font-family:'Martian Mono'; font-size:10px; color:var(--faint); margin-top:7px; }

/* Empty state */
.empty { padding:30px 0 22px; }
.empty h3 { font-size:19px; font-weight:500; color:var(--text); margin:0 0 7px; }
.empty p { color:var(--dim); font-size:14px; margin:0; max-width:56ch; line-height:1.55; }

/* Buttons */
.stButton button {
  background:transparent !important; border:1px solid var(--line) !important;
  color:var(--dim) !important; border-radius:3px !important; font-family:'Archivo' !important;
  font-size:13px !important; font-weight:400 !important; }
.stButton button:hover { border-color:var(--signal) !important; color:var(--text) !important; }
.stButton button:focus-visible { outline:2px solid var(--signal) !important; outline-offset:2px !important; }
.stSpinner > div { border-top-color:var(--signal) !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background:var(--raise) !important; border-color:var(--line) !important;
  color:var(--text) !important; border-radius:3px !important; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

MEDIA_TYPES = {"image", "audio", "video"}

EXAMPLES = [
    "What do all of these sources have in common?",
    "Summarize everything indexed so far",
    "What is shown or said in the media files?",
    "Where do the sources disagree with each other?",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def temp_copy(uploaded):
    """Write an upload to a temp file and remove it once ingestion is done."""
    suffix = "." + uploaded.name.rsplit(".", 1)[-1]
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(uploaded.read())
        tmp.close()
        yield tmp.name
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp.name)


def render_chunks(chunks):
    """Render retrieved context under an answer."""
    if not chunks:
        return
    with st.expander(f"Context used — {len(chunks)} chunks"):
        for c in chunks:
            stype = c["source_type"]
            tag = SOURCE_TAGS.get(stype, "SRC")
            cls = "chunk media" if stype in MEDIA_TYPES else "chunk"
            st.markdown(
                f'<div class="{cls}">{html.escape(c["content"])}'
                f'<div class="from">{tag} · {html.escape(c["source_label"])}</div></div>',
                unsafe_allow_html=True,
            )


def ingest(fn, *args, busy: str, done: str):
    """Run one ingestion call with a spinner, reporting failure in place."""
    with st.spinner(busy):
        try:
            n = fn(*args)
        except Exception as e:
            st.error(f"{e}")
            return False
    st.success(done.format(n=n))
    return True


# ── Session state ──────────────────────────────────────────────────────────────

st.session_state.setdefault("rag", None)
st.session_state.setdefault("api_key", "")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_q", None)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**RAG That Reads,<br>Sees, and Hears**", unsafe_allow_html=True)

    api_key = st.text_input(
        "Google API key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        placeholder="AIza…",
        help="Free key from aistudio.google.com",
    )

    # Rebuild the client whenever the key changes, not only on first entry —
    # otherwise correcting a typo leaves the old, broken client in place.
    if api_key and (st.session_state.rag is None or api_key != st.session_state.api_key):
        st.session_state.rag = MultimodalRAG(api_key=api_key)
        st.session_state.api_key = api_key

    if api_key:
        st.markdown('<p class="state"><i></i>Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="state off"><i></i>Add a key to begin</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Add a source**")

    kind = st.selectbox(
        "Source type",
        ["Text", "URL", "PDF", "Image", "Audio", "Video"],
        label_visibility="collapsed",
    )
    rag: MultimodalRAG | None = st.session_state.rag
    added = False

    if kind == "Text":
        body = st.text_area("Text", placeholder="Paste anything…", height=120,
                            label_visibility="collapsed")
        label = st.text_input("Label", placeholder="Name it, e.g. Meeting notes")
        if st.button("Index text", use_container_width=True, disabled=not (api_key and body.strip())):
            added = ingest(rag.add_text, body.strip(), label or "Pasted text",
                           busy="Indexing…", done="Indexed {n} chunks")

    elif kind == "URL":
        url = st.text_input("URL", placeholder="https://example.com/article",
                            label_visibility="collapsed")
        if st.button("Fetch and index", use_container_width=True, disabled=not (api_key and url.strip())):
            added = ingest(rag.add_url, url.strip(),
                           busy="Fetching the page…", done="Indexed {n} chunks")

    elif kind == "PDF":
        pdf = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
        if pdf and pdf.size / 1e6 > 10:
            st.caption(f"{pdf.size / 1e6:.1f} MB. Only the first 100 pages are indexed.")
        if st.button("Index PDF", use_container_width=True, disabled=not (api_key and pdf)):
            with temp_copy(pdf) as path:
                added = ingest(rag.add_pdf, path,
                               busy="Reading the PDF…", done="Indexed {n} chunks")

    elif kind == "Image":
        img = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "gif"],
                               label_visibility="collapsed")
        if st.button("Describe and index", use_container_width=True, disabled=not (api_key and img)):
            with temp_copy(img) as path:
                added = ingest(rag.add_image, path,
                               busy="Uploading, then describing the image…",
                               done="Indexed {n} chunks")

    elif kind == "Audio":
        aud = st.file_uploader("Audio", type=["mp3", "wav", "ogg", "m4a", "flac"],
                               label_visibility="collapsed")
        if st.button("Transcribe and index", use_container_width=True, disabled=not (api_key and aud)):
            with temp_copy(aud) as path:
                added = ingest(rag.add_audio, path,
                               busy="Uploading, then transcribing. Long clips take a while…",
                               done="Indexed {n} chunks")

    else:
        vid = st.file_uploader("Video", type=["mp4", "mov", "webm", "avi"],
                               label_visibility="collapsed")
        if st.button("Watch and index", use_container_width=True, disabled=not (api_key and vid)):
            with temp_copy(vid) as path:
                added = ingest(rag.add_video, path,
                               busy="Uploading, then watching. Long clips take a while…",
                               done="Indexed {n} chunks")

    if added:
        st.rerun()

    st.divider()

    # ── The register ──
    if rag and rag.sources:
        st.markdown(f'<p class="reg-head">Indexed — {rag.source_count()} sources, '
                    f'{rag.chunk_count()} chunks</p>', unsafe_allow_html=True)
        for s in rag.sources:
            label = s.label if len(s.label) <= 34 else s.label[:31] + "…"
            cls = "row media" if s.source_type in MEDIA_TYPES else "row"
            st.markdown(
                f'<div class="{cls}"><span class="tag">{SOURCE_TAGS.get(s.source_type, "SRC")}</span>'
                f'<span class="lab">{html.escape(label)}</span>'
                f'<span class="cnt">{s.chunks}</span></div>',
                unsafe_allow_html=True,
            )
        if rag.media_count():
            st.markdown('<p class="legend">Sources marked in <b>teal</b> keep their original '
                        'file. Gemini reopens those when they come back in a search.</p>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<p class="reg-head">Nothing indexed yet.</p>', unsafe_allow_html=True)

    st.divider()

    c1, c2 = st.columns(2)
    if c1.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if c2.button("Start over", use_container_width=True):
        st.session_state.rag = MultimodalRAG(api_key=api_key) if api_key else None
        st.session_state.messages = []
        st.rerun()


# ── Main column ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="mast">
  <h1>One index, six kinds of source</h1>
  <p>Text, web pages, PDFs, images, audio, and video go into the same store.
  Ask a question and the answer is drawn from whichever of them is relevant,
  with the sources named.</p>
</div>
""", unsafe_allow_html=True)

if rag and rag.sources:
    st.markdown(
        f'<div class="strip">'
        f'<div class="stat"><div class="n">{rag.source_count()}</div><div class="k">sources</div></div>'
        f'<div class="stat"><div class="n">{rag.chunk_count()}</div><div class="k">chunks</div></div>'
        f'<div class="stat"><div class="n">{rag.media_count()}</div><div class="k">files Gemini can reopen</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="turn-q">{html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(msg["content"])
        render_chunks(msg.get("chunks"))

if not st.session_state.messages:
    st.markdown("""
    <div class="empty">
      <h3>Ask across everything at once</h3>
      <p>Add sources on the left, then ask below. When an image, audio file, or
      video is relevant, the file itself is sent to the model, not just the
      description written when you indexed it.</p>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, use_container_width=True, key=f"ex{i}"):
            st.session_state.pending_q = q

# ── Ask ────────────────────────────────────────────────────────────────────────

prompt = st.session_state.pending_q or st.chat_input("Ask about your sources…")
st.session_state.pending_q = None

if prompt:
    if not api_key:
        st.error("Add your Google API key on the left first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="turn-q">{html.escape(prompt)}</div>', unsafe_allow_html=True)

    with st.spinner("Searching, then answering…"):
        try:
            result = st.session_state.rag.query(prompt)
            st.markdown(result.answer)

            chunks = [
                {
                    "content": d.page_content,
                    "source_type": d.metadata.get("source_type", "unknown"),
                    "source_label": d.metadata.get("source_label", "unknown"),
                }
                for d in result.retrieved_docs
            ]
            render_chunks(chunks)
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "chunks": chunks}
            )
        except Exception as e:
            msg = f"That query didn't complete: {e}"
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
