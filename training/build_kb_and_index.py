import pickle
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import nltk
import re

nltk.download('punkt')

def clean(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_knowledge_base(limit=3000):
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{limit}]")

    sentences = []

    for item in dataset:
        article = clean(item["article"])
        sents = nltk.sent_tokenize(article)

        for s in sents:
            if len(s) > 30:
                sentences.append(s)

    return sentences


def build_index(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(
        documents,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


if __name__ == "__main__":
    print("Building knowledge base...")
    documents = build_knowledge_base(limit=3000)

    print("Building FAISS index...")
    index, embeddings = build_index(documents)

    print("Saving artifacts...")
    faiss.write_index(index, "saved/faiss.index")

    with open("saved/documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    print("✅ Training complete. Saved in /saved/")