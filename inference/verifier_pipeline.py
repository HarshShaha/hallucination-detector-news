import pickle
import faiss
import numpy as np
import torch
import nltk

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
nltk.download('punkt')


class HallucinationDetector:
    def __init__(self):
        import pickle
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from rank_bm25 import BM25Okapi

        print("Loading saved index...")

        self.index = faiss.read_index("../saved/faiss.index")

        with open("../saved/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        # ✅ BM25 initialization (FIX)
        print("Initializing BM25...")
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # ✅ Embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # ✅ NLI model
        print("Loading NLI model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-3")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "valhalla/distilbart-mnli-12-3"
        ).to(self.device)

        self.model.eval()

        print("✅ Initialization complete")

    def extract_claims(self, text):
        return nltk.sent_tokenize(text)

    def retrieve(self, query, top_k=15, final_k=5):
        # """
        # Generalized retrieval (no hardcoding):
        # - Hybrid BM25 + Dense
        # - Semantic similarity filtering
        # - Robust ranking
        # """

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # -------------------------
        # 1. BM25 retrieval
        bm25_scores = self.bm25.get_scores(query_words)
        bm25_top_idx = np.argsort(bm25_scores)[-top_k:]

        # -------------------------
        # 2. Dense retrieval
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        distances, dense_top_idx = self.index.search(query_vec, top_k)

        dense_top_idx = dense_top_idx[0]
        dense_scores = -distances[0]

        # -------------------------
        # 3. Combine candidates
        candidate_indices = list(set(bm25_top_idx.tolist() + dense_top_idx.tolist()))

        results = []

        for idx in candidate_indices:
            doc = self.documents[idx]

            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())

            # -------------------------
            # Keyword overlap
            overlap = len(query_words & doc_words)

            # -------------------------
            # Dense similarity
            if idx in dense_top_idx:
                dense_score = dense_scores[list(dense_top_idx).index(idx)]
            else:
                dense_score = 0

            # -------------------------
            # BM25 score
            bm25_score = bm25_scores[idx]

            # -------------------------
            # FINAL SCORE
            final_score = (
                0.5 * bm25_score +
                0.4 * dense_score +
                0.1 * overlap
            )

            # -------------------------
            # 🔥 GENERAL FILTER (IMPORTANT)
            if overlap < 2:
                continue

            # 🔥 SEMANTIC FILTER (KEY FIX)
            if dense_score < -1.5:   # threshold (tune this)
                continue

            results.append((doc, final_score))

        # -------------------------
        # Fallback if too strict
        if len(results) < 3:
            for idx in candidate_indices:
                doc = self.documents[idx]
                results.append((doc, bm25_scores[idx]))

        # -------------------------
        # Sort results
        results.sort(key=lambda x: x[1], reverse=True)

        # -------------------------
        # Return top results
        final_docs = [r[0] for r in results[:final_k]]

        return final_docs

    def verify(self, claims, evidences):
        inputs = self.tokenizer(
            evidences,
            claims,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        labels = ["contradiction", "neutral", "entailment"]

        results = []
        for prob in probs:
            scores = {labels[i]: prob[i].item() for i in range(3)}
            results.append(scores)

        return results

    def classify(self, scores):
        ent = max([s["entailment"] for s in scores])
        con = max([s["contradiction"] for s in scores])
        if ent > 0.6:
            return "FACTUAL", ent
        elif con > 0.6:
            return "HALLUCINATED", con
        else:
            return "UNCERTAIN", max(ent, con)

    def analyze(self, text):
        claims = self.extract_claims(text)

        results = []

        for claim in claims:
            evidences = self.retrieve(claim)

            scores = self.verify(
                [claim] * len(evidences),
                evidences
            )

            label, conf = self.classify(scores)

            results.append({
                "claim": claim,
                "result": label,
                "confidence": round(conf, 3),
                "evidence": evidences
            })

        return results