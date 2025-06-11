import numpy as np
from retrieval_modules.bm25 import BM25Retriever
from retrieval_modules.dpr import DPRRetriever
from retrieval_modules.gbr import GraphRetriever
from retrieval_modules.memory import MemoryRetriever

class MultiAgentRetriever:
    def __init__(self, chunk_index):
        self.chunk_index = chunk_index  # Dict[filename -> List[chunk]]
        self.all_chunks = [chunk for chunks in chunk_index.values() for chunk in chunks]

        self.agents = [
            BM25Retriever(),
            DPRRetriever(),
            GraphRetriever(),
            MemoryRetriever()
        ]

    def retrieve(self, query, filename=None, top_k=5):
        # Select relevant chunks
        chunks = self.chunk_index.get(filename) if filename else self.all_chunks

        if not chunks:
            print(f"[WARN] No chunks found for filename: {filename}")
            return []

        results = []
        for agent in self.agents:
            try:
                agent_results = agent.retrieve(query, chunks, top_k=top_k)
                results.extend(agent_results)
            except Exception as e:
                print(f"[ERROR] {agent.__class__.__name__} failed: {e}")

        # Deduplicate and rank by score
        seen = set()
        unique_results = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            if r["text"] not in seen:
                unique_results.append(r)
                seen.add(r["text"])

        return unique_results[:top_k]
