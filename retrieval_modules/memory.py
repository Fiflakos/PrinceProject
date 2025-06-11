# retrieval_modules/memory.py

from typing import List, Dict


class MemoryStore:
    def __init__(self):
        self.memory = []

    def store(self, documents: List[Dict]):
        self.memory.extend(documents)

    def clear(self):
        self.memory = []

    def get_recent(self, n: int = 5) -> List[Dict]:
        return self.memory[-n:]

    def search_in_memory(self, query: str) -> List[Dict]:
        return [doc for doc in self.memory if query.lower() in doc.get("text", "").lower()]
