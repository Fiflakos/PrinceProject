# retrieval_modules/reranker.py

from typing import List, Dict, Any
from transformers import pipeline

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.reranker = pipeline("text-classification", model=model_name, return_all_scores=True)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []

        pairs = [{"text": query, "text_pair": doc["text"]} for doc in documents]
        scores = self.reranker(pairs)

        for doc, score_output in zip(documents, scores):
            # Assuming binary classification, pick positive class score
            doc["score"] = score_output[1]["score"] if len(score_output) > 1 else score_output[0]["score"]

        sorted_docs = sorted(documents, key=lambda x: x["score"], reverse=True)
        return sorted_docs[:top_k]
