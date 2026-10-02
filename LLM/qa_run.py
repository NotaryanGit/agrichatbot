import os
import numpy as np
import argparse
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from run_embedding import load_text_docs, get_embeddings
except Exception:
    # fallback simple loader
    def load_text_docs(data_dir='data/text_files'):
        from types import SimpleNamespace
        return [SimpleNamespace(page_content='This is a sample document about attention mechanisms in transformers. Attention allows models to focus on relevant parts of input.')]
    def get_embeddings(texts):
        rng = np.random.RandomState(0)
        return rng.rand(len(texts), 384)


def build_index(docs):
    texts = [d.page_content for d in docs]
    embeddings = get_embeddings(texts).astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms
    return texts, embeddings, normed


def query_index(query, texts, embeddings, normed, top_k=3):
    q_emb = get_embeddings([query]).astype('float32')
    q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    q_normed = q_emb / q_norm
    sims = (normed @ q_normed.T).ravel()
    idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[int(i)]), texts[int(i)]) for i in idx]


def generate_answer_with_openai(query, contexts):
    try:
        import openai
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            return None
        openai.api_key = key
        prompt = f"Answer the question: {query}\n\nContext:\n{contexts}\n\nAnswer:" 
        resp = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{"role":"user","content":prompt}], max_tokens=256)
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', type=str, default='What is attention mechanism?')
    parser.add_argument('--top_k', type=int, default=3)
    args = parser.parse_args()

    docs = load_text_docs()
    if not docs:
        print('[WARN] No documents found; creating a sample doc.')
        from types import SimpleNamespace
        docs = [SimpleNamespace(page_content='Attention lets neural networks focus on relevant parts of the input when producing each output, enabling transformers to weigh input tokens differently.')]

    texts, embeddings, normed = build_index(docs)
    results = query_index(args.query, texts, embeddings, normed, top_k=args.top_k)

    contexts = "\n\n".join([f"- {r[2]}" for r in results])
    print('[INFO] Top contexts:')
    for i, sim, txt in results:
        print(f"  - (score={sim:.4f}) {txt[:200]}")

    answer = generate_answer_with_openai(args.query, contexts)
    if answer:
        print('\n[LLM Answer via OpenAI]')
        print(answer)
    else:
        print('\n[Fallback Answer]')
        print('Based on retrieved context:')
        print(contexts)

if __name__ == '__main__':
    main()
