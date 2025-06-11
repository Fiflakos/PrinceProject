class BM25Retriever:
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.index = self.build_index()

    def build_index(self):
        pass
