"""
KrishiMitra Answer Generation Module Alias
"""
from answer_genration import (
    generate_answer,
    retrieve_documents,
    build_rag_prompt,
    call_groq_llm,
    get_vector_store,
    get_embedding_model,
    get_groq_client,
)

__all__ = [
    "generate_answer",
    "retrieve_documents",
    "build_rag_prompt",
    "call_groq_llm",
    "get_vector_store",
    "get_embedding_model",
    "get_groq_client",
]

if __name__ == "__main__":
    from answer_genration import __name__ as _, main
    pass
