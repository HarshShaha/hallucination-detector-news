import pickle
import faiss
import numpy as np
import torch
import nltk

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

nltk.download('punkt')


class HallucinationDetector:

    def __init__(self):
        print("Loading saved index...")

        self.index = faiss.read_index("saved/faiss.index")

        with open("saved/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        print("Loading NLI model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-3")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "valhalla/distilbart-mnli-12-3"
        ).to(self.device)

        self.model.eval()

    def extract_claims(self, text):
        return nltk.sent_tokenize(text)

    def retrieve(self, claim, top_k=3):
        query_vec = self.embedder.encode([claim], convert_to_numpy=True)
        _, indices = self.index.search(query_vec, top_k)

        return [self.documents[i] for i in indices[0]]

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

        if con > 0.6:
            return "HALLUCINATED", con
        elif ent > 0.6:
            return "FACTUAL", ent
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