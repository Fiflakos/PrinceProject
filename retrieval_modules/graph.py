# retrieval_modules/graph.py

from typing import List, Dict
from collections import defaultdict
import difflib


class GraphConnector:
    def __init__(self):
        self.graph = defaultdict(set)

    def build_graph(self, documents: List[Dict]):
        for i, doc in enumerate(documents):
            meta_i = doc.get("meta", {})
            for j, other_doc in enumerate(documents):
                if i == j:
                    continue
                meta_j = other_doc.get("meta", {})
                if meta_i.get("author") == meta_j.get("author"):
                    self.graph[i].add(j)
                elif self._similar_title(doc["text"], other_doc["text"]):
                    self.graph[i].add(j)

    def get_related_documents(self, index: int, documents: List[Dict]) -> List[Dict]:
        related_indices = self.graph.get(index, set())
        return [documents[i] for i in related_indices]

    def _similar_title(self, text1: str, text2: str) -> bool:
        return difflib.SequenceMatcher(None, text1[:100], text2[:100]).ratio() > 0.8
