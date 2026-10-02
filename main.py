"""
KrishiMitra Agricultural Chatbot Entrypoint
--------------------------------------------
1. Ensures FAISS vector database exists (runs ingestion if needed).
2. Runs interactive multi-turn agricultural chat loop powered by Groq LLM + RAG.
"""

import sys
import os
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_PATH = Path("db/faiss_index")

def ensure_vector_db():
    """Verify that vector database exists; if not, trigger ingestion pipeline."""
    if not DB_PATH.exists() or not (DB_PATH / "index.faiss").exists():
        print("🌱 Vector store not found – running ingestion pipeline now...")
        from ingestion_pipeline import run_ingestion
        run_ingestion()
        print("✅ Ingestion complete.\n")


def start_chat():
    ensure_vector_db()
    
    from answer_genration import generate_answer
    
    history = []
    print("=" * 65)
    print("🌾 Welcome to KrishiMitra - AI Agriculture Assistant 🌾")
    print("Ask any question about crops, yields, seasons, and farming!")
    print("Type 'quit' or 'exit' to exit.")
    print("=" * 65)
    
    while True:
        try:
            query = input("\n🧑‍🌾 You: ").strip()
            if not query:
                continue
            if query.lower() in {"quit", "exit", "q"}:
                print("\n🙏 Thank you for using KrishiMitra. Happy farming! 🚜🌾\n")
                break
                
            print("⏳ KrishiMitra is thinking...")
            result = generate_answer(query, history)
            
            print(f"\n🤖 KrishiMitra:\n{result['answer']}")
            
            if result.get("sources"):
                print("\n📚 Data Sources:")
                for idx, s in enumerate(result["sources"], 1):
                    crop_info = f" | Crop: {s['crop']}" if s.get('crop') and s['crop'] != 'N/A' else ""
                    state_info = f" | State: {s['state']}" if s.get('state') and s['state'] != 'N/A' else ""
                    print(f"  [{idx}] {s['source']}{crop_info}{state_info}")
                    
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": result["answer"]})
            
        except KeyboardInterrupt:
            print("\n\n🙏 Session ended. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    start_chat()
