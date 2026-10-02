import os
import sys
import warnings
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

# Reconfigure stdout to handle Unicode/Hindi characters on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ---------- CONFIGURATION ----------
PERSIST_DIRECTORY = Path(os.getenv("PERSIST_DIRECTORY", "db/faiss_index"))
EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
FALLBACK_GROQ_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

SYSTEM_PROMPT = """You are KrishiMitra, an expert and trustworthy agricultural AI assistant for Indian farmers.
Answer the user's question accurately and helpfully.

Guidelines:
1. Use simple, practical, and farmer-friendly language. Avoid unnecessary technical jargon.
2. If agricultural context data is provided below, summarize and analyze it clearly to answer the user's question.
3. If the question is a general agricultural query (e.g. general farming practices, pest control, soil care, or crop cultivation tips), you are encouraged to use your pre-trained agricultural knowledge to provide a comprehensive and helpful response, incorporating the provided context if relevant.
4. If the question asks for specific localized statistics (such as crop yield, production, area, or weather details in a specific year, state, or district) and there is NOT enough relevant data in the provided context, politely state:
   "I don't have reliable database statistics for that specific query in my current records. Please consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer."
5. Do NOT make up crop yields, fertilizer dosages, or scheme eligibility criteria.
"""

# ---------- LAZY INITIALIZATION ----------
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[FAISS] = None
_groq_client: Optional[Groq] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def get_vector_store() -> FAISS:
    global _vector_store
    if _vector_store is None:
        if not PERSIST_DIRECTORY.exists() or not (PERSIST_DIRECTORY / "index.faiss").exists():
            print(f"[KrishiMitra] Vector index not found at '{PERSIST_DIRECTORY}'. Running ingestion pipeline...")
            from ingestion_pipeline import run_ingestion
            run_ingestion()

        embeddings = get_embedding_model()
        _vector_store = FAISS.load_local(
            str(PERSIST_DIRECTORY),
            embeddings,
            allow_dangerous_deserialization=True
        )
    return _vector_store


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in environment or .env file. "
                "Please add GROQ_API_KEY=your_key in .env"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


_dataframes: Optional[Dict[str, pd.DataFrame]] = None

def get_dataframes() -> Dict[str, pd.DataFrame]:
    global _dataframes
    if _dataframes is None:
        _dataframes = {}
        data_dir = Path("data/raw")
        csv_files = glob.glob(str(data_dir / "*.csv"))
        
        # If no CSVs are found in data/raw, check if we need to unzip
        if not csv_files:
            print("[KrishiMitra] CSV files not found in data/raw. Checking zip archives...")
            from ingestion_pipeline import unzip_all, ARCHIVE_DIR, EXTRACT_DIR
            try:
                csv_files = unzip_all(ARCHIVE_DIR, EXTRACT_DIR)
            except Exception as e:
                print(f"[Warning] Failed to unzip data: {e}")
                csv_files = []
                
        for f in csv_files:
            name = Path(f).name
            try:
                df = pd.read_csv(f, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(f, encoding="latin-1")
            _dataframes[name] = df
            print(f"[KrishiMitra] Loaded dataset '{name}' with {len(df):,} rows.")
    return _dataframes


def extract_filters_from_query(query: str) -> Dict[str, Any]:
    """Query Groq to parse agricultural queries into structured search filters (JSON)."""
    try:
        client = get_groq_client()
        EXTRACT_PROMPT = """You are an AI assistant designed to extract search parameters from a user's agricultural question.
Extract the following fields in JSON format:
- state: (str or null) the Indian state name mentioned (e.g. Haryana, Punjab, Tamil Nadu)
- district: (str or null) the district name mentioned (e.g. Karnal, Moga, Erode)
- crop: (str or null) the crop name mentioned (e.g. Rice, Wheat, Tomato)
- season: (str or null) the season (Kharif, Rabi, Summer, Whole Year)
- year: (int or null) the year if a specific year is requested
- is_general_knowledge: (bool) set to true if the question is a general agricultural question (e.g., how to grow a crop, pest control, fertilizing, organic farming) and does NOT ask for specific local statistical data (like yield, production, weather, area in a specific district/state/year).

Response format: ONLY return a valid JSON object.
"""
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": f"User question: {query}"}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[Warning] Groq filter extraction unavailable ({e}). Using offline extractor.")
        try:
            from offline_chatbot import extract_filters_offline
            return extract_filters_offline(query)
        except Exception:
            return {
                "state": None,
                "district": None,
                "crop": None,
                "season": None,
                "year": None,
                "is_general_knowledge": True
            }


def retrieve_structured_documents(filters: Dict[str, Any], k: int = TOP_K) -> List[Any]:
    """Search across all full CSV datasets using extracted filters and return structured results."""
    from langchain_core.documents import Document
    
    dfs = get_dataframes()
    results = []
    
    state = filters.get("state")
    district = filters.get("district")
    crop = filters.get("crop")
    season = filters.get("season")
    year = filters.get("year")
    
    # If no structured filters are provided, return empty
    if not any([state, district, crop, season, year]):
        return []
        
    for name, df in dfs.items():
        # Identify columns
        state_col = next((c for c in df.columns if c.lower() in ["state", "state_name", "state_names", "state name"]), None)
        crop_col = next((c for c in df.columns if c.lower() in ["crop", "crop_name", "crop_names", "crop_type"]), None)
        season_col = next((c for c in df.columns if c.lower() in ["season", "season_name", "season_names"]), None)
        district_col = next((c for c in df.columns if c.lower() in ["district", "district_name", "district_names", "dist name"]), None)
        year_col = next((c for c in df.columns if c.lower() in ["year", "crop_year", "crop year"]), None)
        
        filtered_df = df
        
        # Apply string matching filters
        if state and state_col:
            filtered_df = filtered_df[filtered_df[state_col].astype(str).str.lower().str.contains(state.lower(), na=False)]
        if district and district_col:
            filtered_df = filtered_df[filtered_df[district_col].astype(str).str.lower().str.contains(district.lower(), na=False)]
        if crop and crop_col:
            filtered_df = filtered_df[filtered_df[crop_col].astype(str).str.lower().str.contains(crop.lower(), na=False)]
        if season and season_col:
            filtered_df = filtered_df[filtered_df[season_col].astype(str).str.lower().str.contains(season.lower(), na=False)]
        if year and year_col:
            filtered_df = filtered_df[filtered_df[year_col].astype(str).str.contains(str(year), na=False)]
            
        if len(filtered_df) > 0:
            # Format row content
            # Limit the matching rows per dataset to prevent context window overflow
            matched_rows = filtered_df.head(k)
            for _, row in matched_rows.iterrows():
                content_parts = []
                for col, val in row.items():
                    if col in ("Unnamed: 0", "index") or pd.isna(val) or val == "":
                        continue
                    content_parts.append(f"{col}: {val}")
                page_content = " | ".join(content_parts)
                
                meta = {
                    "source": name,
                    "state": str(row[state_col]) if state_col and not pd.isna(row[state_col]) else "N/A",
                    "crop": str(row[crop_col]) if crop_col and not pd.isna(row[crop_col]) else "N/A"
                }
                results.append(Document(page_content=page_content, metadata=meta))
                
    return results[:k]


# ---------- CORE RETRIEVAL & GENERATION ----------
def retrieve_documents(query: str, k: int = TOP_K) -> List[Any]:
    """Retrieve relevant document chunks using FAISS similarity search."""
    vs = get_vector_store()
    return vs.similarity_search(query, k=k)


def build_rag_prompt(query: str, docs: List[Any], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Construct the prompt with conversation history and retrieved context."""
    context_blocks = []
    for doc in docs:
        source_name = doc.metadata.get("source", "Agricultural Dataset")
        state = doc.metadata.get("state")
        crop = doc.metadata.get("crop")
        extra = []
        if state:
            extra.append(f"State: {state}")
        if crop:
            extra.append(f"Crop: {crop}")
        header = f"[Source: {source_name}" + (f" | {', '.join(extra)}" if extra else "") + "]"
        context_blocks.append(f"{header}\n{doc.page_content}")

    context_str = "\n\n".join(context_blocks)

    history_str = ""
    if chat_history:
        recent = chat_history[-6:]  # last 3 conversational turns
        history_str = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in recent)

    prompt = f"""Conversation History:
{history_str if history_str else "No previous conversation."}

Agricultural Context Data:
{context_str}

Farmer Question: {query}

Helpful Answer:"""
    return prompt


def call_groq_llm(prompt: str) -> str:
    """Query Groq API with Llama-3.3."""
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=600,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# ---------- PUBLIC API ----------
def generate_answer(query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Generate an answer for the user query using RAG.
    
    Returns:
        dict: {
            "answer": str,
            "sources": List[dict],
            "used_context": bool
        }
    """
    if chat_history is None:
        chat_history = []

    # Extract search filters from query
    filters = extract_filters_from_query(query)
    
    docs = []
    # Only retrieve documents if it's not a pure general knowledge query without state/district/crop
    if not (filters.get("is_general_knowledge") and not any([filters.get("state"), filters.get("district"), filters.get("crop")])):
        # Try structured Pandas search first
        docs = retrieve_structured_documents(filters, k=TOP_K)
        
        # Fallback to FAISS vector store if structured search returned nothing
        if not docs:
            try:
                vs = get_vector_store()
                docs = vs.similarity_search(query, k=TOP_K)
            except Exception as e:
                print(f"[Warning] FAISS similarity search failed: {e}")
                docs = []

    # If it is a specific statistical query (not general knowledge) and no documents were retrieved,
    # return the statistical fallback answer directly.
    if not docs and not filters.get("is_general_knowledge", True):
        return {
            "answer": (
                "I don't have reliable database statistics for that specific query in my current records. "
                "Please consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer."
            ),
            "sources": [],
            "used_context": False,
        }

    if os.getenv("OFFLINE_MODE", "false").lower() in ("true", "1", "yes"):
        from offline_chatbot import generate_offline_answer
        return generate_offline_answer(query, chat_history)

    # Build prompt and call LLM
    try:
        prompt = build_rag_prompt(query, docs, chat_history)
        answer = call_groq_llm(prompt)
    except Exception as e:
        print(f"[Warning] Groq LLM failed ({e}). Falling back to Offline Engine...")
        from offline_chatbot import generate_offline_answer
        return generate_offline_answer(query, chat_history)

    sources = [
        {
            "source": d.metadata.get("source", "Agricultural Dataset"),
            "state": d.metadata.get("state", "N/A"),
            "crop": d.metadata.get("crop", "N/A"),
            "preview": d.page_content[:150]
        }
        for d in docs
    ]

    return {
        "answer": answer,
        "sources": sources,
        "used_context": len(docs) > 0,
    }


# ---------- CLI TEST INTERFACE ----------
if __name__ == "__main__":
    print("\n🌱 KrishiMitra RAG Engine initialized.")
    print("Type your agriculture question below (or 'quit' / 'exit' to stop):\n")
    history: List[Dict[str, str]] = []
    
    while True:
        try:
            q = input("Farmer: ").strip()
            if not q:
                continue
            if q.lower() in {"quit", "exit"}:
                print("\nGoodbye! Happy farming! 🌾")
                break
                
            res = generate_answer(q, history)
            print(f"\nKrishiMitra: {res['answer']}\n")
            
            if res.get("sources"):
                print("📚 Sources retrieved:")
                for s in res["sources"]:
                    print(f"  - {s['source']} (Crop: {s.get('crop', 'N/A')}, State: {s.get('state', 'N/A')})")
                print()
                
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": res["answer"]})
            
        except KeyboardInterrupt:
            print("\nSession ended. 🌾")
            break
        except Exception as err:
            print(f"\n[Error] {err}\n")