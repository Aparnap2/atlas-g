# 🛠️ Atlas-G — Field Ops Assistant

> Stateful, context-aware AI agent for field technicians. Retrieves precise repair procedures from complex PDF manuals and cross-references real-time inventory — enabling **first-time fix** success.

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Atlas-G Agent System                   │
│                                                         │
│  [STT: Whisper]  →  [LangGraph Orchestrator]            │
│                         │                               │
│            ┌────────────┴────────────┐                  │
│            ▼                         ▼                  │
│   [Technical_Expert Node]   [Inventory_Checker Node]    │
│   (Docling RAG + Granite)   (Mock DB Tool Call)         │
│            │                         │                  │
│            └──────────┬──────────────┘                  │
│                       ▼                                 │
│              [Redis Checkpoint]                         │
│                       │                                 │
│                       ▼                                 │
│             [TTS: Kokoro Output]                        │
└─────────────────────────────────────────────────────────┘
```

## 🗂️ Project Structure

```
atlas-g/
├── ingest/
│   └── docling_parser.py       # PDF → chunks via Docling
├── agent/
│   ├── state.py                # AgentState (TypedDict)
│   ├── tools.py                # Inventory tool + RAG retriever
│   ├── nodes.py                # Technical_Expert & Inventory_Checker nodes
│   └── graph.py                # LangGraph stateful workflow
├── memory/
│   └── redis_checkpoint.py     # Redis checkpointer setup
├── voice/
│   ├── tts_kokoro.py           # Kokoro TTS output
│   └── stt_whisper.py          # Whisper STT input
├── data/
│   └── mock_inventory.json     # Mock inventory database
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example
└── docker-compose.yml          # Redis service
```

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| PDF Parsing | [Docling](https://github.com/docling-project/docling) (IBM) |
| LLM | IBM Granite 3.1 via [Ollama](https://ollama.com) |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| State/Memory | [Redis](https://redis.io) + `langgraph-checkpoint-redis` |
| TTS | [Kokoro-82M](https://github.com/hexgrad/kokoro) (local, 82M params) |
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (local) |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector Store | FAISS (in-memory, local) |

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Ollama and pull models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull granite3.1-dense:8b
ollama pull nomic-embed-text
```

### 2. Setup

```bash
git clone https://github.com/Aparnap2/atlas-g.git
cd atlas-g
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Start Redis

```bash
docker-compose up -d
```

### 4. Ingest a PDF Manual

```bash
python main.py ingest --pdf /path/to/service_manual.pdf
```

### 5. Run the Agent

```bash
# Text mode
python main.py chat --thread-id field-tech-001

# Voice mode (Whisper STT + Kokoro TTS)
python main.py voice --thread-id field-tech-001
```

## 🔄 State Recovery Demo

```bash
# Session 1: Start a conversation
python main.py chat --thread-id tech-van-007
> "Error code E-44 on the hydraulic press, what are the steps?"

# [Simulate crash / device switch]

# Session 2: Resume seamlessly from Redis checkpoint
python main.py chat --thread-id tech-van-007
> "Did you find the gasket part number?"
# Agent remembers full context from Session 1 ✅
```

## 📊 KPIs

| Metric | Target |
|---|---|
| Table part-number retrieval accuracy | 100% |
| Redis state recovery success rate | 100% |
| End-to-end reasoning latency | < 5 seconds |

## 📄 License

MIT License — Built with ❤️ for field technicians everywhere.
