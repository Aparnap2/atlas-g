"""Atlas-G — Streamlit Field Ops Assistant UI."""
import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Atlas-G — Field Ops Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark industrial theme */
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
}
.atlas-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border: 1px solid #00d4ff33;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.atlas-header h1 { color: #00d4ff; margin: 0; font-size: 2rem; }
.atlas-header p  { color: #8892a4; margin: 0.3rem 0 0; font-size: 0.9rem; }

/* Chat bubbles */
.user-bubble {
    background: #1e2d3d;
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #e0e6f0;
}
.agent-bubble {
    background: #1a2218;
    border-left: 3px solid #39d353;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #e0e6f0;
}
.inv-badge-ok  { background:#1a3a1a; color:#39d353; border-radius:4px; padding:2px 8px; font-size:0.8rem; }
.inv-badge-out { background:#3a1a1a; color:#ff6b6b; border-radius:4px; padding:2px 8px; font-size:0.8rem; }
.inv-badge-wh  { background:#1a2a3a; color:#ffd700; border-radius:4px; padding:2px 8px; font-size:0.8rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1f2937;
}
.thread-tag {
    background: #00d4ff22;
    border: 1px solid #00d4ff44;
    border-radius: 6px;
    padding: 4px 10px;
    color: #00d4ff;
    font-family: monospace;
    font-size: 0.85rem;
}
.status-dot-green { color: #39d353; }
.status-dot-red   { color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)


# ── Lazy-load heavy modules (cached for the session) ─────────────────
@st.cache_resource(show_spinner="🔗 Connecting to Redis...")
def _load_checkpointer():
    from memory.redis_checkpoint import get_checkpointer
    return get_checkpointer()


@st.cache_resource(show_spinner="🧠 Building Atlas-G graph...")
def _load_graph(_checkpointer):
    from agent.graph import build_graph
    return build_graph(checkpointer=_checkpointer)


def get_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛠️ Atlas-G")
    st.markdown("**Field Ops Assistant**")
    st.divider()

    # ─ Session ─
    st.markdown("### 📍 Session")
    thread_id = st.text_input(
        "Thread ID",
        value=st.session_state.get("thread_id", os.getenv("DEFAULT_THREAD_ID", "tech-van-001")),
        help="Reuse a Thread ID to restore a previous conversation from Redis.",
    )
    if st.button("Load / Resume Session", use_container_width=True):
        st.session_state["thread_id"] = thread_id
        st.session_state["messages"] = []
        st.session_state["session_loaded"] = False
        st.rerun()

    if "thread_id" in st.session_state:
        st.markdown(f'<span class="thread-tag">🔗 {st.session_state["thread_id"]}</span>', unsafe_allow_html=True)

    st.divider()

    # ─ PDF Ingest ─
    st.markdown("### 📄 Ingest Manual")
    uploaded_pdf = st.file_uploader(
        "Upload PDF Service Manual",
        type=["pdf"],
        help="Parsed with IBM Docling — tables, headers, and error codes preserved.",
    )
    if uploaded_pdf and st.button("▶️ Ingest PDF", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name
        with st.spinner(f"Docling parsing {uploaded_pdf.name}..."):
            try:
                from ingest.docling_parser import ingest_pdf
                count = ingest_pdf(tmp_path)
                st.success(f"✔ {count} chunks indexed from {uploaded_pdf.name}")
            except Exception as e:
                st.error(f"Ingest failed: {e}")
        Path(tmp_path).unlink(missing_ok=True)

    st.divider()

    # ─ Voice ─
    st.markdown("### 🔊 Voice Settings")
    tts_enabled = st.toggle("Kokoro TTS", value=False, help="Speak agent responses aloud.")
    voice_choice = st.selectbox(
        "Voice",
        ["af_heart", "af_nova", "am_adam", "am_echo", "bf_emma", "bm_george"],
        index=0,
        disabled=not tts_enabled,
    )
    tts_speed = st.slider("Speed", 0.7, 1.5, 1.0, 0.05, disabled=not tts_enabled)

    st.divider()

    # ─ Model Info ─
    st.markdown("### ⚙️ Stack")
    st.markdown("""
    | Layer | Model |
    |---|---|
    | LLM | `granite-docling:latest` |
    | Embed | `nomic-embed-text` |
    | Parse | IBM Docling |
    | Memory | Redis |
    | TTS | Kokoro-82M |
    | STT | faster-whisper |
    """)

    st.divider()
    if st.button("🗑️ Clear Chat Display", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ── Session State Init ─────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = os.getenv("DEFAULT_THREAD_ID", "tech-van-001")
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "session_loaded" not in st.session_state:
    st.session_state["session_loaded"] = False
if "inventory_panels" not in st.session_state:
    st.session_state["inventory_panels"] = []


# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="atlas-header">
  <h1>▲ ATLAS-G</h1>
  <p>Field Ops Assistant — IBM Granite-Docling · LangGraph · Redis · Kokoro TTS · Whisper STT</p>
</div>
""", unsafe_allow_html=True)


# ── Load graph + checkpointer ───────────────────────────────────────
checkpointer = _load_checkpointer()
graph = _load_graph(checkpointer)
config = get_thread_config(st.session_state["thread_id"])


# ── Restore Redis session on first load ─────────────────────────────
if not st.session_state["session_loaded"]:
    try:
        saved = graph.get_state(config)
        if saved and saved.values.get("messages"):
            saved_msgs = saved.values["messages"]
            st.session_state["messages"] = [
                {"role": "assistant" if isinstance(m, AIMessage) else "user", "content": m.content}
                for m in saved_msgs
            ]
            st.toast(
                f"✔ Resumed — {len(saved_msgs)} messages restored from Redis",
                icon="🔗",
            )
    except Exception:
        pass
    st.session_state["session_loaded"] = True


# ── Status bar ───────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Thread", st.session_state["thread_id"])
with col2:
    st.metric("Messages", len(st.session_state["messages"]))
with col3:
    vs_path = Path(os.getenv("VECTOR_STORE_PATH", "./vector_store"))
    st.metric("Vector Store", "✔ Ready" if vs_path.exists() else "✗ Empty")
with col4:
    st.metric("LLM", "granite-docling:latest")

st.divider()


# ── Chat Display ──────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state["messages"]:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color: #4a5568;">
            <div style="font-size:3rem">🛠️</div>
            <div style="font-size:1.1rem; margin-top:1rem;">Atlas-G is ready.</div>
            <div style="font-size:0.85rem; margin-top:0.5rem;">Upload a PDF manual in the sidebar, then ask a repair question.</div>
            <div style="font-size:0.8rem; margin-top:1rem; color:#2d3748;">
                Example: <i>"Error code E-44 on the hydraulic press — what are the repair steps?"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👷"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🛠️"):
                    st.markdown(msg["content"])

    # Inventory result panels (shown inline after agent response)
    for panel in st.session_state.get("inventory_panels", []):
        with st.expander("📦 Inventory Check Results", expanded=True):
            for item in panel:
                result_text = item.get("result", "")
                if "IN STOCK" in result_text:
                    badge = '<span class="inv-badge-ok">✔ IN STOCK</span>'
                elif "OUT OF STOCK" in result_text or "ZERO" in result_text:
                    badge = '<span class="inv-badge-out">✖ OUT OF STOCK</span>'
                else:
                    badge = '<span class="inv-badge-wh">⚠ CHECK WAREHOUSE</span>'
                st.markdown(
                    f"{badge} &nbsp; `{item['args'].get('part_number', '')}` — {result_text}",
                    unsafe_allow_html=True,
                )


# ── Chat Input ────────────────────────────────────────────────────
if prompt := st.chat_input("🔧  Describe the fault or ask a repair question..."):
    # Add user message immediately
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👷"):
        st.markdown(prompt)

    # Run the agent
    with st.chat_message("assistant", avatar="🛠️"):
        with st.spinner("⚡ Atlas-G reasoning via granite-docling..."):
            t0 = time.time()
            try:
                inputs = {"messages": [HumanMessage(content=prompt)]}
                result = graph.invoke(inputs, config=config)
                latency = time.time() - t0

                messages = result.get("messages", [])
                last_ai = next(
                    (m for m in reversed(messages)
                     if isinstance(m, AIMessage)),
                    None,
                )
                answer = last_ai.content if last_ai else "No response generated."

                st.markdown(answer)
                st.caption(f"⏱ {latency:.2f}s · granite-docling:latest · thread: {st.session_state['thread_id']}")

                # TTS
                if tts_enabled:
                    try:
                        from voice.tts_kokoro import speak
                        speak(answer, voice=voice_choice, speed=tts_speed)
                    except Exception as tts_err:
                        st.warning(f"TTS error: {tts_err}")

                # Inventory panel
                inv = result.get("inventory_results", [])
                if inv:
                    st.session_state["inventory_panels"].append(inv)
                    with st.expander("📦 Inventory Check Results", expanded=True):
                        for item in inv:
                            result_text = item.get("result", "")
                            if "IN STOCK" in result_text:
                                badge = '<span class="inv-badge-ok">✔ IN STOCK</span>'
                            elif "OUT OF STOCK" in result_text or "ZERO" in result_text:
                                badge = '<span class="inv-badge-out">✖ OUT OF STOCK</span>'
                            else:
                                badge = '<span class="inv-badge-wh">⚠ CHECK WAREHOUSE</span>'
                            st.markdown(
                                f"{badge} &nbsp; `{item['args'].get('part_number', '')}` — {result_text}",
                                unsafe_allow_html=True,
                            )

                st.session_state["messages"].append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Agent error: {e}")
                st.session_state["messages"].append(
                    {"role": "assistant", "content": f"⚠️ Error: {e}"}
                )
