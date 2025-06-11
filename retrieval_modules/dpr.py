# retrieval_modules/dpr.py

from typing import List, Dict
from sentence_transformers import SentenceTransformer
import os
import faiss
import numpy as np


class DPRRetriever:
    def __init__(self, data_dir: str = "data_cleaned"):
        self.model = SentenceTransformer("facebook-dpr-question_encoder-single-nq-base")
        self.documents = []
        self.index = None
        self._load_and_index_documents(data_dir)

    def _load_and_index_documents(self, data_dir: str):
        texts = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                        self.documents.append({"id": path, "text": text})
                        texts.append(text)

        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            doc = self.documents[idx].copy()
            doc["score"] = float(dist)
            results.append(doc)

        return results
