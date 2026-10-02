# 🌱 KrishiMitra (Agri-Pulse) — Intelligent Agricultural AI Assistant

An advanced, production-grade **Retrieval-Augmented Generation (RAG)** agricultural advisory system designed to empower Indian farmers, agronomists, and researchers with real-time, data-backed insights on crop selection, yield prediction, soil health, and weather patterns.

Supports **both Online Cloud LLMs (Groq)** and **100% Offline Local Execution (Ollama / Local Synthesizer)**.

---

## 🌟 Key Features

- **🌾 Domain-Specific Agricultural Knowledge Base**:
  - Ingests and indexes historical Indian crop production data, state-wise soil characteristics (NPK & pH), rainfall, and weather patterns.
  - Multi-dataset synthesis covering state, district, crop types, season, area, and yields across 600+ districts.

- **⚡ Blazing Fast Retrieval & Dual-Mode Inference**:
  - High-performance vector embeddings with local `sentence-transformers/all-MiniLM-L6-v2`.
  - Low-latency similarity search using **FAISS** (`faiss-cpu`) running locally on CPU.
  - **Online Mode**: Ultra-fast responses via **Groq** (`qwen/qwen3.8-27b`).
  - **100% Offline Mode**: Runs with **zero internet** via [offline_chatbot.py](file:///c:/Users/Admin/OneDrive/Desktop/agri%20chatbot/offline_chatbot.py) (using local Ollama or built-in offline knowledge synthesizer).

- **🖥️ Full-Stack Interactive Interface**:
  - **FastAPI Backend**: Asynchronous REST API serving chat endpoints, source attribution, and document streaming (`server.py`).
  - **Modern Web Dashboard**: Glassmorphic, animated UI (`preview.html`) featuring voice input (Speech-to-Text), interactive 3D elements, suggested prompts, and citation cards.

---

## 🏗️ Architecture Overview

```
                          [User Query] 
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   [Online Mode: server.py]              [Offline Mode: offline_chatbot.py]
            │                                     │
            ▼                                     ▼
[Local Sentence-Transformers]          [Local Regex & Parameter Parser]
            │                                     │
            ▼                                     ▼
[Local FAISS Vector Index (db/)]       [Local CSV Datasets (data/raw)]
            │                                     │
            └──────────────────┬──────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     (Internet Available)                 (Zero Internet / Offline)
            │                                     │
            ▼                                     ▼
     [Groq Cloud LLM]                     [Local Ollama or Synthesizer]
            │                                     │
            └──────────────────┬──────────────────┘
                               │
                               ▼
        [Structured Agricultural Answer with Source Citations]
```

---

## 📁 Repository Structure

```
├── answer_generation.py     # Module alias for answer generation
├── answer_genration.py      # Core RAG logic: retriever, Groq client, prompt templates
├── offline_chatbot.py       # 100% Offline execution engine & CLI (Zero internet required)
├── ingestion_pipeline.py    # Pipeline to process raw CSVs, chunk text, and build FAISS index
├── server.py                # FastAPI REST API server
├── main.py                  # CLI / Interactive execution entry point
├── preview.html             # Full-featured modern web interface (Frontend)
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
- (Optional for online mode): Free [Groq API Key](https://console.groq.com)
- (Optional for offline neural LLM): [Ollama](https://ollama.com)

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
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

---

## 🌐 Running Online (FastAPI + Web UI)

Start the FastAPI server:
```bash
python server.py
```
Or with uvicorn:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in your browser, or open `preview.html` directly.

---

## 📴 Running 100% Offline (No Internet / No APIs)

KrishiMitra includes native offline execution that works on local datasets and FAISS indices without needing an internet connection.

### Method 1: Interactive Offline Terminal (Instant)
Run the dedicated offline assistant directly:
```bash
python offline_chatbot.py
```
You can also ask single questions directly from the command line:
```bash
python offline_chatbot.py --query "What is the yield of Rice in Haryana during Kharif season?"
python offline_chatbot.py --query "What are effective organic pest management methods for vegetables?"
```

### Method 2: Launch the Offline Web Server & UI
Launch the local web server in offline mode:
```bash
python offline_chatbot.py --server
```
Visit **`http://localhost:8000`** in your browser to use the full visual dashboard offline.

### Method 3: Enhanced Offline Mode with Local Ollama (Optional)
If you want local neural LLM reasoning offline:
1. Download Ollama from [ollama.com](https://ollama.com).
2. Pull and run a local model:
   ```bash
   ollama run llama3.2
   # or
   ollama run qwen2.5:3b
   ```
3. Run `python offline_chatbot.py` — it will automatically detect and route prompts through your local Ollama server!

---

## 🧪 Verification & Testing
Run the automated verification suite:
```bash
python verify_chatbot.py
```
This tests preset queries (yield queries, crop suitability, organic pest control, district data, and out-of-scope filtering).

---

## 🛡️ License & Acknowledgements
- Developed for smart agriculture and farmer empowerment.
- Datasets sourced from open Indian agricultural repositories and Kaggle.
- Powered by [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), and [Groq Cloud](https://groq.com/).
