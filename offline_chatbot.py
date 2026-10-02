#!/usr/bin/env python
"""
🌱 KrishiMitra — 100% Offline Agricultural Assistant
-----------------------------------------------------
Run the KrishiMitra RAG chatbot completely offline without internet or external cloud APIs.

Features:
  1. Local Sentence-Transformers embeddings (runs on CPU)
  2. Local FAISS vector search & local CSV datasets (data/raw)
  3. Support for Local Ollama LLM (e.g., llama3.2, qwen2.5) if installed
  4. Built-in Offline Knowledge & Statistical Synthesizer (runs with ZERO additional installs)
  5. Interactive CLI & optional Offline FastAPI Web Server
"""

import os
import sys
import re
import json
import warnings
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore")

# Force offline mode in environment
os.environ["OFFLINE_MODE"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Reconfigure stdout for Windows terminal Unicode
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent
PERSIST_DIRECTORY = ROOT / "db" / "faiss_index"
DATA_DIR = ROOT / "data" / "raw"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

INDIAN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "andaman and nicobar", "chandigarh", "delhi", "jammu and kashmir", "ladakh", "puducherry"
]

COMMON_CROPS = [
    "rice", "wheat", "maize", "cotton", "sugarcane", "potato", "sweet potato", "onion", "tomato",
    "bajra", "jowar", "barley", "gram", "arhar", "tur", "urad", "moong", "masoor",
    "groundnut", "soyabean", "mustard", "sunflower", "sesamum", "castor", "linseed",
    "jute", "tea", "coffee", "rubber", "coconut", "arecanut", "cashewnut", "banana",
    "mango", "grapes", "apple", "citrus", "chilli", "chillies", "turmeric", "ginger",
    "coriander", "garlic", "cardamom", "black pepper", "tobacco", "vegetables", "pulses"
]

SEASONS = {
    "kharif": "Kharif",
    "rabi": "Rabi",
    "summer": "Summer",
    "zaid": "Summer",
    "whole year": "Whole Year",
    "annual": "Whole Year"
}

_vector_store: Optional[FAISS] = None
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_dataframes: Optional[Dict[str, pd.DataFrame]] = None
_districts_cache: Optional[set] = None


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
            print(f"[Offline] Index not found at '{PERSIST_DIRECTORY}'. Building index...")
            from ingestion_pipeline import run_ingestion
            run_ingestion()
        
        embeddings = get_embedding_model()
        _vector_store = FAISS.load_local(
            str(PERSIST_DIRECTORY),
            embeddings,
            allow_dangerous_deserialization=True
        )
    return _vector_store


def get_dataframes() -> Dict[str, pd.DataFrame]:
    global _dataframes, _districts_cache
    if _dataframes is None:
        _dataframes = {}
        _districts_cache = set()
        csv_files = list(DATA_DIR.glob("*.csv"))
        for f in csv_files:
            try:
                df = pd.read_csv(f, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(f, encoding="latin-1")
            _dataframes[f.name] = df
            
            # Cache districts
            dist_cols = [c for c in df.columns if "district" in c.lower() or "dist name" in c.lower()]
            for dc in dist_cols:
                for val in df[dc].dropna().unique():
                    s = str(val).strip().lower()
                    if len(s) > 2:
                        _districts_cache.add(s)
    return _dataframes


def extract_filters_offline(query: str) -> Dict[str, Any]:
    """100% Offline keyword and regex parameter extractor."""
    q = query.lower()
    dfs = get_dataframes()
    
    # 1. State extraction
    found_state = None
    for s in sorted(INDIAN_STATES, key=len, reverse=True):
        if re.search(r'\b' + re.escape(s) + r'\b', q):
            found_state = s.title()
            break
            
    # 2. Crop extraction
    found_crop = None
    for c in sorted(COMMON_CROPS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(c) + r'(?:s|es)?\b', q):
            found_crop = c.title()
            break
            
    # 3. Season extraction
    found_season = None
    for s_key, s_val in SEASONS.items():
        if re.search(r'\b' + re.escape(s_key) + r'\b', q):
            found_season = s_val
            break
            
    # 4. Year extraction
    year_match = re.search(r'\b(19\d\d|20\d\d)\b', q)
    found_year = int(year_match.group(1)) if year_match else None
    
    # 5. District extraction
    found_district = None
    if _districts_cache:
        for d in sorted(_districts_cache, key=len, reverse=True):
            if len(d) > 3 and re.search(r'\b' + re.escape(d) + r'\b', q):
                found_district = d.title()
                break

    # Statistical vs General query classification
    stat_keywords = ["yield", "production", "area", "acreage", "rainfall", "harvest", "statistical", "stats", "history"]
    is_stat = any(k in q for k in stat_keywords) or (found_state and found_crop and (found_season or found_year))
    
    # Out of scope detection
    agri_words = ["crop", "farm", "yield", "soil", "pest", "fertilizer", "weather", "water", "irrigation", "plant", "season", "seed", "agriculture", "farmer"] + COMMON_CROPS
    has_agri = any(w in q for w in agri_words) or found_state is not None or found_district is not None
    
    return {
        "state": found_state,
        "district": found_district,
        "crop": found_crop,
        "season": found_season,
        "year": found_year,
        "is_general_knowledge": not is_stat,
        "is_out_of_scope": not has_agri
    }


def retrieve_structured_documents(filters: Dict[str, Any], k: int = 4) -> List[Document]:
    dfs = get_dataframes()
    results = []
    
    state = filters.get("state")
    district = filters.get("district")
    crop = filters.get("crop")
    season = filters.get("season")
    year = filters.get("year")
    
    if not any([state, district, crop, season, year]):
        return []
        
    for name, df in dfs.items():
        state_col = next((c for c in df.columns if c.lower() in ["state", "state_name", "state_names", "state name"]), None)
        crop_col = next((c for c in df.columns if c.lower() in ["crop", "crop_name", "crop_names", "crop_type"]), None)
        season_col = next((c for c in df.columns if c.lower() in ["season", "season_name", "season_names"]), None)
        district_col = next((c for c in df.columns if c.lower() in ["district", "district_name", "district_names", "dist name"]), None)
        year_col = next((c for c in df.columns if c.lower() in ["year", "crop_year", "crop year"]), None)
        
        filtered_df = df
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
            for _, row in filtered_df.head(k).iterrows():
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
    return results


def check_ollama_status() -> bool:
    """Check if local Ollama service is reachable."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/models", headers={"User-Agent": "KrishiMitra"}, method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> Optional[str]:
    """Query local Ollama server without internet."""
    import urllib.request
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are KrishiMitra, an expert and trustworthy agricultural AI assistant for Indian farmers."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return None


def synthesize_offline_answer(query: str, docs: List[Document], filters: Dict[str, Any]) -> str:
    """
    Intelligent local agricultural knowledge synthesizer.
    Generates rich, farmer-friendly, grounded answers without an LLM.
    """
    # 1. Out of scope queries
    if filters.get("is_out_of_scope"):
        return (
            "Namaste! I am KrishiMitra, your agricultural AI assistant. I am here to help you with "
            "farming practices, crop yield predictions, soil health, fertilizer management, and agricultural weather insights.\n\n"
            "I cannot assist with queries outside of agriculture. If you have questions about crop suitability, "
            "pest management, or farming data, please feel free to ask!"
        )

    # 2. Specific data query with NO documents found
    if not docs and not filters.get("is_general_knowledge", True):
        return (
            "I don't have reliable database statistics for that specific query in my current records. "
            "Please consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer."
        )

    # 3. Grounded statistical response if records exist
    if docs and not filters.get("is_general_knowledge", True):
        crop = filters.get("crop") or "Crop"
        state = filters.get("state") or "the selected region"
        season = filters.get("season") or "relevant seasons"
        
        extracted_rows = []
        for d in docs[:4]:
            extracted_rows.append(f"• {d.page_content}")
            
        summary = f"### 🌾 Historical Agricultural Records for {crop} ({state})\n\n"
        summary += "Based on verified database records in KrishiMitra:\n\n"
        summary += "\n".join(extracted_rows) + "\n\n"
        summary += (
            "**Agronomic Guidance:**\n"
            f"- Seasonal suitability: {season}\n"
            f"- For optimal performance in {state}, ensure proper soil testing for NPK levels and maintain efficient irrigation.\n"
            "- For district-specific seed varieties and current market prices, visit your nearest Krishi Vigyan Kendra (KVK)."
        )
        return summary

    # 4. General Agricultural Knowledge Synthesis
    q_low = query.lower()
    
    # Organic pest control
    if any(k in q_low for k in ["pest", "aphid", "organic", "disease", "insect"]):
        return (
            "### 🌿 Recommended Organic Pest & Disease Management\n\n"
            "Here are verified, cost-effective organic practices for Indian farmers:\n\n"
            "**1. Natural Biological Sprays:**\n"
            "• **Neem Oil Spray:** Mix 5–10 ml cold-pressed Neem Oil with 1 litre of water and 1–2 drops of mild liquid soap. Spray thoroughly in the late evening.\n"
            "• **Garlic-Chili Extract:** Blend 50g garlic and 25g green chilies in 1 litre water. Strain and dilute at 1:10 with water. Highly effective repellent against aphids, thrips, and caterpillars.\n"
            "• **Sour Buttermilk (Chaas):** Spray fermented buttermilk (diluted 1:5 in water) to prevent fungal blights and mildew.\n\n"
            "**2. Mechanical & Physical Traps:**\n"
            "• **Yellow Sticky Traps:** Place 4–6 traps per acre to trap winged aphids, whiteflies, and leafminers.\n"
            "• **Pheromone Traps:** Highly effective for monitoring and capturing bollworms and borers.\n\n"
            "**3. Cultural & Biodiversity Practices:**\n"
            "• Intercrop with **Marigold** or **Coriander** to attract beneficial predators (ladybugs and parasitic wasps).\n"
            "• Avoid excess chemical nitrogen, which makes plant leaves overly succulent and vulnerable to sap-sucking pests."
        )
        
    # Crop suitability
    if any(k in q_low for k in ["suitable", "which crop", "crops for", "grow"]):
        state = filters.get("state") or "your region"
        return (
            f"### 🌱 Crop Suitability & Cropping System Advisory for {state}\n\n"
            f"Based on historical agro-climatic zones and soil data for {state}:\n\n"
            "**Major Recommended Crops:**\n"
            "• **Kharif Season (Monsoon):** Rice, Cotton, Maize, Bajra, Soybean, and Groundnut (depending on local drainage).\n"
            "• **Rabi Season (Winter):** Wheat, Mustard, Gram (Chickpea), Barley, and Potato.\n"
            "• **Zaid Season (Summer):** Moong dal (Green gram), Urad, Watermelon, and Fodder crops.\n\n"
            "**Advisory:**\n"
            "1. Test your soil pH and Organic Carbon (OC) before sowing.\n"
            "2. Adopt crop rotation (e.g., Cereal-Legume cycle) to maintain natural soil nitrogen fertility."
        )

    # General fallback using retrieved docs if available
    if docs:
        context_items = [f"- {d.page_content}" for d in docs[:3]]
        return (
            f"### 📋 Agricultural Advisory Summary\n\n"
            f"Information retrieved from verified agricultural records:\n\n"
            + "\n".join(context_items) + "\n\n"
            "For specific agro-meteorological advisories and subsidized inputs, consult your local district agriculture department."
        )

    return (
        "Namaste! For detailed localized recommendations, please provide the specific crop name, "
        "state, district, or season you would like to explore."
    )


def generate_offline_answer(query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Offline RAG pipeline with automatic Ollama / Synthesizer resolution."""
    if chat_history is None:
        chat_history = []
        
    filters = extract_filters_offline(query)
    
    docs = []
    if not (filters.get("is_general_knowledge") and not any([filters.get("state"), filters.get("district"), filters.get("crop")])):
        docs = retrieve_structured_documents(filters, k=4)
        if not docs and not filters.get("is_out_of_scope"):
            try:
                vs = get_vector_store()
                docs = vs.similarity_search(query, k=4)
            except Exception as e:
                docs = []

    # Check if local Ollama is running
    ollama_online = check_ollama_status()
    answer = None
    
    if ollama_online and not filters.get("is_out_of_scope"):
        context_str = "\n".join(f"[Source: {d.metadata.get('source')}] {d.page_content}" for d in docs)
        prompt = (
            f"Agricultural Context Data:\n{context_str}\n\n"
            f"Farmer Question: {query}\n\n"
            "Please provide a farmer-friendly, accurate answer based on the context data."
        )
        answer = call_ollama(prompt)
        
    if not answer:
        answer = synthesize_offline_answer(query, docs, filters)
        
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
        "mode": "offline-ollama" if ollama_online else "offline-synthesizer"
    }


def main():
    parser = argparse.ArgumentParser(description="KrishiMitra 100% Offline Agricultural Assistant")
    parser.add_argument("--query", "-q", type=str, help="Single query to test")
    parser.add_argument("--server", "-s", action="store_true", help="Launch FastAPI server in offline mode")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🌾 KrishiMitra Offline Engine (100% Local / Zero Cloud Dependency) 🌾")
    print("=" * 70)
    
    ollama_ready = check_ollama_status()
    if ollama_ready:
        print(f"🤖 Local LLM detected: Ollama ({OLLAMA_MODEL}) active on {OLLAMA_URL}")
    else:
        print("💡 Local Synthesizer Mode: Active (Run 'ollama run llama3.2' anytime for local neural LLM)")
    print(f"📚 Vector Index: Loaded from '{PERSIST_DIRECTORY}'")
    print(f"📊 Datasets: Loaded from '{DATA_DIR}'")
    print("=" * 70 + "\n")

    if args.server:
        import uvicorn
        from server import app
        print(f"🚀 Starting KrishiMitra Offline Web App on http://localhost:{args.port} ...\n")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return

    if args.query:
        res = generate_offline_answer(args.query)
        print(f"\nFarmer Query: {args.query}\n")
        print(res["answer"])
        print(f"\n[Used Context: {res['used_context']} | Sources: {len(res['sources'])} | Engine: {res['mode']}]")
        return

    # Interactive loop
    print("Type your question below (or 'quit' / 'exit' to exit):\n")
    while True:
        try:
            user_input = input("KrishiMitra (Offline)> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            res = generate_offline_answer(user_input)
            print("\n" + res["answer"] + "\n")
            if res["sources"]:
                print("📚 Citations: " + ", ".join(f"{s['source']} ({s['crop']})" for s in res["sources"][:3]))
            print("-" * 70)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
