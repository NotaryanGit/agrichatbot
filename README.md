# 🌱 KrishiMitra (Agri-Pulse) — Intelligent Agricultural AI Assistant

An advanced, production-grade **Retrieval-Augmented Generation (RAG)** agricultural advisory system designed to empower Indian farmers, agronomists, and researchers with real-time, data-backed insights on crop selection, yield prediction, soil health, and weather patterns.

---

## 🌟 Key Features

- **🌾 Domain-Specific RAG Knowledge Base**:
  - Ingests and indexes historical Indian crop production data, state-wise soil characteristics, rainfall, and weather patterns.
  - Multi-dataset synthesis covering state, district, crop types, season, and yields.

- **⚡ Blazing Fast Retrieval & Inference**:
  - High-performance vector embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
  - Local, low-latency similarity search using **FAISS** (`faiss-cpu`).
  - Ultra-fast LLM responses via **Groq** (`llama-3.3-70b-versatile` / `gemma2-9b-it`).

- **🖥️ Full-Stack Interactive Interface**:
  - **FastAPI Backend**: Asynchronous REST API serving chat endpoints, source attribution, and document streaming.
  - **Modern Web Dashboard**: Intuitive, glassmorphic UI (`preview.html`) featuring voice input support, quick prompt suggestions, conversation history, and source breakdown cards.

---

## 🏗️ Architecture Overview

```
[User Query] 
     │
     ▼
[FastAPI Server (server.py)]
     │
     ▼
[Embed Query: Sentence-Transformers (all-MiniLM-L6-v2)]
     │
     ▼
[FAISS Vector Store (db/faiss_index)] ──> Retrieve Top-K Context Chunks
     │
     ▼
[Prompt Engineering (State/Crop Context + Advisory System Prompt)]
     │
     ▼
[Groq LLM Engine (Llama 3.3 / Gemma)]
     │
     ▼
[Structured Response with Citations & Source Attribution]
```

---

## 📁 Repository Structure

```
├── answer_generation.py     # Module alias for answer generation
├── answer_genration.py      # Core RAG logic: retriever, Groq client, prompt templates
├── ingestion_pipeline.py    # Pipeline to process raw CSVs, chunk text, and build FAISS index
├── server.py                # FastAPI REST API server
├── main.py                  # CLI / Interactive execution entry point
├── preview.html             # Full-featured web interface (Frontend)
├── verify_chatbot.py        # Automated test verification suite
├── requirements.txt         # Python project dependencies
├── .env.example             # Template for required environment variables
├── .gitignore               # Ignored files (secrets, venvs, cache)
├── data/
│   └── raw/                 # Cleaned datasets (crop yield, soil, weather data)
├── db/
│   └── faiss_index/         # Pre-built FAISS vector store index & metadata
└── LLM/                     # Research notebooks, documentation, and exploratory scripts
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- A free [Groq API Key](https://console.groq.com)

### 2. Clone the Repository
```bash
git clone https://github.com/NotaryanGit/agrichatbot.git
cd agrichatbot
```

### 3. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your Groq API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

## 🛠️ Running the Application

### Option A: Web Interface & API (Recommended)
Start the FastAPI server:
```bash
python server.py
```
Or with uvicorn directly:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in your browser, or open `preview.html` directly.

### Option B: CLI Interactive Mode
Run the console-based assistant:
```bash
python main.py
```

### Option C: Rebuilding Vector Index (Optional)
To ingest new data or rebuild the FAISS vector database from scratch:
```bash
python ingestion_pipeline.py
```

---

## 🧪 Verification & Testing
Run the automated verification script to validate index loading, retrieval, and LLM inference:
```bash
python verify_chatbot.py
```

---

## 🛡️ License & Acknowledgements
- Developed for smart agriculture and farmer empowerment.
- Datasets sourced from open agricultural repositories and Kaggle India agriculture databases.
- Powered by [LangChain](https://www.langchain.com/) and [Groq Cloud](https://groq.com/).
