# 🛠️ Atlas-G — Field Ops Assistant

> **Stateful, context-aware AI agent for field service technicians.**
> Retrieves precise repair procedures from complex PDF manuals and cross-references real-time inventory — enabling **first-time fix** success. 100% local, 100% open-source.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet?logo=astral)](https://docs.astral.sh/uv)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![Docling](https://img.shields.io/badge/parsing-IBM%20Docling-0062ff)](https://github.com/docling-project/docling)
[![Redis](https://img.shields.io/badge/memory-Redis-red?logo=redis)](https://redis.io)
[![Kokoro TTS](https://img.shields.io/badge/TTS-Kokoro--82M-green)](https://github.com/hexgrad/kokoro)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 📑 Table of Contents

1. [Product Vision](#-product-vision)
2. [High-Level Design (HLD)](#-high-level-design-hld)
3. [Low-Level Design (LLD)](#-low-level-design-lld)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Quick Start with uv](#-quick-start-with-uv)
7. [Streamlit UI](#-streamlit-ui)
8. [State Recovery Demo](#-state-recovery-demo)
9. [KPIs](#-kpis)
10. [License](#-license)

---

## 🎯 Product Vision

Field service technicians work under extreme time pressure — manually searching 5,000-page industrial PDFs while simultaneously checking legacy inventory systems for parts. **Atlas-G** eliminates that friction by combining:

- **IBM Docling** for high-fidelity PDF parsing (tables, error codes, torque specs preserved exactly)
- **granite-docling:latest (258M)** via Ollama as the local reasoning engine
- **LangGraph** for stateful multi-agent orchestration
- **Redis checkpoints** for session continuity across device switches and connectivity loss
- **Kokoro TTS + Whisper STT** for hands-free voice interaction

---

## 🏗️ High-Level Design (HLD)

The system is composed of four loosely-coupled subsystems that communicate through the shared `AgentState`.

```
╔══════════════════════════════════════════════════════════════════╗
║                     ATLAS-G  SYSTEM BOUNDARY                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌──────────────┐    ┌───────────────────────────────────────┐  ║
║  │  INPUT LAYER │    │         ORCHESTRATION LAYER            │  ║
║  │              │    │                                       │  ║
║  │ Streamlit UI │───▶│  LangGraph StateGraph                 │  ║
║  │  (app.py)    │    │                                       │  ║
║  │              │    │  ┌─────────────────────────────────┐  │  ║
║  │ Whisper STT  │    │  │  Node: technical_expert         │  │  ║
║  │ (voice mode) │    │  │  RAG over Docling-parsed FAISS  │  │  ║
║  └──────────────┘    │  │  → extracts steps + part nums   │  │  ║
║                       │  └──────────────┬──────────────────┘  │  ║
║  ┌──────────────┐    │                 │  (if parts found)    │  ║
║  │  DATA LAYER  │    │  ┌──────────────▼──────────────────┐  │  ║
║  │              │    │  │  Node: inventory_checker         │  │  ║
║  │ Docling PDF  │    │  │  Tool-calling: van + warehouse   │  │  ║
║  │  Parser      │───▶│  │  → check_inventory()            │  │  ║
║  │              │    │  │  → search_warehouse_b()         │  │  ║
║  │ FAISS Vector │    │  └─────────────────────────────────┘  │  ║
║  │  Store       │    └───────────────────────────────────────┘  ║
║  └──────────────┘                      │                        ║
║                                        ▼                        ║
║  ┌──────────────┐    ┌───────────────────────────────────────┐  ║
║  │ OUTPUT LAYER │    │         MEMORY LAYER                   │  ║
║  │              │    │                                       │  ║
║  │ Kokoro TTS   │◀───│  Redis RedisSaver Checkpointer        │  ║
║  │ Streamlit UI │    │  key: thread_id → full AgentState     │  ║
║  └──────────────┘    │  survives: crashes, device switches   │  ║
║                       └───────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════╝
```

### HLD Component Responsibilities

| Component | Responsibility | Technology |
|---|---|---|
| **Input Layer** | Accept text or voice queries from the technician | Streamlit, faster-whisper |
| **Data Layer** | Parse PDFs into structured chunks; serve relevant context | IBM Docling, FAISS, nomic-embed-text |
| **Orchestration Layer** | Route query through agent nodes; call tools; synthesise answer | LangGraph, granite-docling:latest (Ollama) |
| **Memory Layer** | Persist full conversation state keyed by `thread_id` | Redis, langgraph-checkpoint-redis |
| **Output Layer** | Return text response + optional spoken TTS audio | Streamlit, Kokoro-82M |

### HLD Data-Flow (happy path)

```
Technician query
      │
      ▼
[Streamlit / CLI]  ──── HumanMessage ────▶  [LangGraph graph.invoke()]
                                                      │
                              ┌───────────────────────┘
                              ▼
                   [technical_expert_node]
                   │  1. Embed query via nomic-embed-text
                   │  2. FAISS similarity search (k=6)
                   │  3. granite-docling:latest generates answer
                   │  4. Regex-extract PART_NUMBERS[]
                   │  5. Set next_action = inventory_checker
                              │
                              ▼
                   [inventory_checker_node]
                   │  1. bind_tools([check_inventory, search_warehouse_b])
                   │  2. LLM emits tool_calls
                   │  3. Execute tools against mock_inventory.json
                   │  4. LLM synthesises natural-language report
                              │
                              ▼
                   [Redis RedisSaver]
                   │  Checkpoint saved after every super-step
                              │
                              ▼
                   [Streamlit response + optional Kokoro TTS]
```

---

## 🔬 Low-Level Design (LLD)

### Module Map

```
atlas-g/
├── app.py                        # Streamlit UI
├── main.py                       # Typer CLI (ingest / chat / voice)
├── pyproject.toml                # uv project manifest + dependencies
├── .python-version               # Python 3.11 pin for uv
├── .env.example                  # Environment variable template
├── docker-compose.yml            # Redis service
│
├── ingest/
│   └── docling_parser.py         # PDF → LangChain Documents → FAISS
│
├── agent/
│   ├── state.py                  # AgentState TypedDict
│   ├── tools.py                  # check_inventory, search_warehouse_b, get_retriever
│   ├── nodes.py                  # technical_expert_node, inventory_checker_node
│   └── graph.py                  # build_graph() — compiles StateGraph
│
├── memory/
│   └── redis_checkpoint.py       # get_checkpointer(), get_thread_config()
│
├── voice/
│   ├── tts_kokoro.py             # speak(), save_to_file()
│   └── stt_whisper.py            # listen(), _transcribe_file()
│
└── data/
    └── mock_inventory.json       # Van stock + Warehouse B mock DB
```

---

### `agent/state.py` — AgentState

```python
class AgentState(TypedDict):
    messages:               Annotated[Sequence[BaseMessage], add_messages]
    identified_part_numbers: list[str]
    inventory_results:      list[dict]
    current_repair_context: Optional[str]
    next_action:            Optional[str]  # "inventory_checker" | "END"
```

`messages` uses LangGraph's `add_messages` reducer — concurrent appends are merged safely. All other fields are last-write-wins.

---

### `ingest/docling_parser.py` — PDF Ingest Pipeline

```
PDF file
  │
  ▼  DocumentConverter (PdfPipelineOptions)
  │   do_ocr=True
  │   do_table_structure=True        ← preserves nested tables
  │   table_structure_options.do_cell_matching=True
  ▼
Docling DoclingDocument
  │
  ├─▶ iterate_items()  → text sections (with page, heading metadata)
  └─▶ doc.tables[]     → export_to_markdown() (each table = 1 Document)
  ▼
RecursiveCharacterTextSplitter  (chunk_size=1000, overlap=150)
  ▼
OllamaEmbeddings (nomic-embed-text)
  ▼
FAISS.save_local("./vector_store")   ← merged if store already exists
```

**Key design decision:** Tables are exported to Markdown and stored as standalone chunks. This guarantees that part numbers inside multi-column tables are never split mid-row by the text splitter.

---

### `agent/nodes.py` — Node LLD

#### `technical_expert_node`

| Step | Detail |
|---|---|
| Input | Latest `HumanMessage` from state |
| Retrieval | `get_retriever()` → FAISS k=6, returns top-6 chunks |
| Prompt | System: `TECHNICAL_SYSTEM_PROMPT` + `{context}` + chat history |
| LLM | `ChatOllama(model="granite-docling:latest", temperature=0.1)` |
| Output parsing | `re.search(r"PART_NUMBERS:\s*\[([^\]]+)\]")` |
| Routing signal | Sets `next_action="inventory_checker"` if parts found |

#### `inventory_checker_node`

| Step | Detail |
|---|---|
| Input | `identified_part_numbers[]` from state |
| Tool binding | `llm.bind_tools([check_inventory, search_warehouse_b])` |
| Pass 1 (LLM) | LLM decides which tools to call and with which args |
| Tool execution | Direct `fn.invoke(tc["args"])` against `mock_inventory.json` |
| Pass 2 (LLM) | Synthesises tool results into technician-friendly language |
| Fallback | If no tool calls emitted, returns LLM's raw content |

---

### `agent/graph.py` — Graph Topology

```
START
  │
  ▼
technical_expert  ──── next_action=="inventory_checker" ────▶ inventory_checker ──▶ END
  │
  └──── next_action=="END" ──────────────────────────────────────────────────────▶ END
```

The graph is compiled once per Streamlit session via `@st.cache_resource`, injected with the Redis `RedisSaver` checkpointer. Every `graph.invoke()` call automatically saves state after each node completes.

---

### `memory/redis_checkpoint.py` — State Persistence

```
thread_id  (e.g. "tech-van-007")
     │
     ▼
RedisSaver.from_conn_string(REDIS_URL)
     │
     ├─ WRITE: after every node completion (automatic by LangGraph)
     └─ READ:  graph.get_state(config)  ← called on Streamlit page load
                       │
                       ▼
               Rehydrates full AgentState:
               - All messages
               - Last repair context
               - Part numbers found so far
               - Inventory results
```

**Crash recovery guarantee:** Because checkpoints are written at sub-graph granularity (after each node), even a mid-session crash only loses the current node's in-flight computation — not the entire conversation.

---

### `voice/` — Voice Pipeline LLD

```
[Whisper STT — stt_whisper.py]
sounddevice.rec(16kHz, 10s)
  → tempfile.wav
  → WhisperModel("base.en", compute_type="int8")  ← ~75MB
  → VAD filter (min_silence_duration_ms=500)
  → transcript string

[Kokoro TTS — tts_kokoro.py]
KPipeline(lang_code="en-us")
  → strip markdown  (re.sub)
  → generator yields (graphemes, phonemes, audio_chunk)
  → np.concatenate(audio_chunks)
  → sounddevice.play(audio, samplerate=24000)
```

---

### `data/mock_inventory.json` — Schema

```json
{
  "Technician_Van_Stock": {
    "<PART_NUMBER>": {
      "description": "string",
      "quantity":    0,
      "unit":        "kit|pcs",
      "bin_location": "VAN-BIN-XX"
    }
  },
  "Warehouse_B": { ... same schema ... }
}
```

The `check_inventory` tool scans **both** locations. If the van is empty, the agent automatically calls `search_warehouse_b` and reports ETA.

---

## ⚙️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| PDF Parsing | [IBM Docling](https://github.com/docling-project/docling) | Best-in-class table structure extraction for industrial manuals |
| LLM | `granite-docling:latest` (258M) via [Ollama](https://ollama.com) | Fine-tuned on document understanding; runs offline on modest hardware |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful, cyclical graphs with conditional routing and checkpointing |
| State / Memory | [Redis](https://redis.io) + `langgraph-checkpoint-redis` | Sub-millisecond reads; survives process crashes |
| UI | [Streamlit](https://streamlit.io) | Rapid field-deployable web UI; `@st.cache_resource` for hot-reloading |
| TTS | [Kokoro-82M](https://github.com/hexgrad/kokoro) | 82M params, 26 voices, runs on CPU, <300ms first-token |
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | INT8-quantised Whisper; ~75MB base.en model |
| Embeddings | `nomic-embed-text` via Ollama | 768-dim, local, no API key |
| Vector Store | FAISS (CPU) | Zero-infra, persistent on disk, merge-able |
| Package Manager | [uv](https://docs.astral.sh/uv) | 10-100× faster than pip; lockfile-first, Python version managed |

---

## 🗂️ Project Structure

```
atlas-g/
├── app.py                    # 🖥️  Streamlit UI (primary interface)
├── main.py                   # 🖥️  CLI fallback (ingest / chat / voice)
├── pyproject.toml            # 📦  uv project manifest
├── .python-version           # 🐍  Python 3.11 pin
├── uv.lock                   # 🔒  Reproducible lockfile (commit this!)
├── .env.example              # 🔑  Environment template
├── docker-compose.yml        # 🐳  Redis service
│
├── ingest/
│   └── docling_parser.py     # PDF → FAISS via Docling
├── agent/
│   ├── state.py              # AgentState TypedDict
│   ├── tools.py              # RAG retriever + inventory tools
│   ├── nodes.py              # technical_expert + inventory_checker
│   └── graph.py              # LangGraph StateGraph
├── memory/
│   └── redis_checkpoint.py   # Redis RedisSaver
├── voice/
│   ├── tts_kokoro.py         # Kokoro TTS
│   └── stt_whisper.py        # Whisper STT
└── data/
    └── mock_inventory.json   # Mock van + warehouse inventory
```

---

## 🚀 Quick Start with uv

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone & sync

```bash
git clone https://github.com/Aparnap2/atlas-g.git
cd atlas-g

# uv reads .python-version (3.11) and pyproject.toml automatically
uv sync
```

> `uv sync` creates `.venv/`, installs all dependencies from `uv.lock` in one shot — no manual `pip install` needed.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (Ollama URL, Redis URL, voice settings)
```

### 4. Pull Ollama models

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull granite-docling:latest
ollama pull nomic-embed-text
```

### 5. Start Redis

```bash
docker-compose up -d
```

### 6. Ingest a PDF manual

```bash
uv run python main.py ingest --pdf /path/to/service_manual.pdf
```

### 7. Launch the Streamlit UI

```bash
uv run streamlit run app.py
# Opens at http://localhost:8501
```

### 8. (Optional) CLI modes

```bash
# Text chat
uv run python main.py chat --thread-id tech-van-001

# Voice mode (Whisper STT + Kokoro TTS)
uv run python main.py voice --thread-id tech-van-001 --duration 10
```

---

## 🖥️ Streamlit UI

| Panel | Feature |
|---|---|
| **Sidebar → Session** | Enter Thread ID → Load / Resume from Redis checkpoint |
| **Sidebar → Ingest Manual** | Drag-drop PDF → Docling parses + indexes to FAISS |
| **Sidebar → Voice Settings** | Toggle Kokoro TTS, pick voice (af_heart, am_adam…), set speed |
| **Status bar** | Live metrics: Thread ID · Message count · Vector Store status · LLM name |
| **Chat window** | `st.chat_message` bubbles with latency caption per response |
| **Inventory expander** | Auto-opens with colour-coded badges: ✔ IN STOCK / ✖ OUT OF STOCK / ⚠ WAREHOUSE |

```bash
uv run streamlit run app.py
```

---

## 🔄 State Recovery Demo

```bash
# ── Session 1 ─────────────────────────────────────────────────────
uv run python main.py chat --thread-id tech-van-007
> Error code E-44 on the hydraulic press — what are the repair steps?
  Atlas-G: Step 1... Step 2... Part needed: HYD-GSKT-4402
  Inventory: Out of stock in van. Found 14 in Warehouse B. Order by 14:00.

# ── Simulate crash / device switch ───────────────────────────────
^C   (or close the laptop and pick up a tablet)

# ── Session 2 ─────────────────────────────────────────────────────
uv run python main.py chat --thread-id tech-van-007
  ✔ Resuming — 6 messages restored from Redis.

> Did you manage to order the gasket?
  Atlas-G: Yes — HYD-GSKT-4402 was identified in our last session.
           Warehouse B has 14 units. Same-day dispatch if ordered before 14:00. ✅
```

---

## 📊 KPIs

| Metric | Target | How measured |
|---|---|---|
| Table part-number retrieval accuracy | 100% | Manual eval against known PDFs |
| Redis state recovery success rate | 100% | Simulated crash test script |
| End-to-end reasoning latency | < 5 s | `time.time()` delta in `app.py` |

---

## 📄 License

MIT License — Built with ❤️ for field technicians everywhere.
