import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

# Import RAG pipeline from answer_generation
from answer_generation import generate_answer

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="KrishiMitra (Agri-Pulse) Backend",
    description="Agricultural AI Assistant API powered by RAG, FAISS, and Groq LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Request / Response Models ----------
class ChatRequest(BaseModel):
    query: str = Field(..., description="Farmer's question")
    history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Conversation history list of {role, content}"
    )

class ChatSource(BaseModel):
    source: str
    state: Optional[str] = "N/A"
    crop: Optional[str] = "N/A"
    preview: Optional[str] = ""

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    used_context: bool = False

# ---------- Endpoints ----------
@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "ok", "service": "KrishiMitra Backend"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    RAG chat endpoint. Receives query and optional chat history,
    queries the FAISS vector database, and generates an answer via Groq.
    """
    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = generate_answer(clean_query, req.history or [])
        return ChatResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            used_context=result.get("used_context", False)
        )
    except Exception as e:
        print(f"[Error in /api/chat] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Serve Frontend ----------
FRONTEND = ROOT / "preview.html"

@app.get("/")
async def serve_root():
    if not FRONTEND.exists():
        raise HTTPException(status_code=404, detail="preview.html not found in project directory")
    return FileResponse(str(FRONTEND))


if __name__ == "__main__":
    import uvicorn
    print("\n🌾 Starting KrishiMitra Agri-Pulse Server on http://localhost:8000 ...\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)