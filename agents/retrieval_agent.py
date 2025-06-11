from retrieval_modules.bm25 import BM25Retriever

class LocalRetrievalAgent:
    def __init__(self, corpus_dir="data_cleaned"):
        self.retriever = BM25Retriever(corpus_dir)

    def retrieve(self, query, top_k=3):
        return self.retriever.retrieve(query, top_k=top_k)
