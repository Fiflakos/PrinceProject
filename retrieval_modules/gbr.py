from typing import List, Dict

class GraphBasedReRanker:
    def __init__(self):
        pass

    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Re-ranks the documents based on simple heuristic for now.
        Replace with a graph-based algorithm later.
        """
        # Dummy logic: Sort by length of content (you can replace this with a real method later)
        ranked = sorted(documents, key=lambda doc: len(doc.get("text", "")), reverse=True)
        return ranked[:5]
