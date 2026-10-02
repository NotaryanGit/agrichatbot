import os
import glob
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except Exception as e:
    HAVE_ST = False
    print(f"[WARN] sentence_transformers not available: {e}; using fake embeddings")

class Doc:
    def __init__(self, text):
        self.page_content = text


def load_text_docs(data_dir='data/text_files'):
    texts = []
    for p in glob.glob(os.path.join(data_dir, '**', '*.txt'), recursive=True):
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(Doc(f.read()))
        except Exception as e:
            print(f"[ERROR] Failed to read {p}: {e}")
    return texts


def get_embeddings(texts):
    if HAVE_ST:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return np.array(model.encode(texts, show_progress_bar=False))
    else:
        rng = np.random.RandomState(0)
        return rng.rand(len(texts), 384)


def main():
    docs = load_text_docs()
    if not docs:
        print('[WARN] No text files found in data/text_files. Creating sample doc.')
        docs = [Doc('This is a sample document for embedding.')]
    texts = [d.page_content for d in docs]
    print(f"[INFO] Generating embeddings for {len(texts)} texts...")
    embeddings = get_embeddings(texts)
    print('[INFO] Embeddings shape:', embeddings.shape)
    print('[INFO] Example embedding:', embeddings[0] if embeddings.shape[0] > 0 else None)


if __name__ == '__main__':
    main()
