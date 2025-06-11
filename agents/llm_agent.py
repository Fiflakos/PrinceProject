from agents.hf_model_generator import LocalHistBERTGenerator
from retrieval_agent import LocalRetrievalAgent


class LocalLLMAgent:
    def __init__(self):
        self.generator = LocalHistBERTGenerator()
        self.retriever = LocalRetrievalAgent()

    def ask(self, question: str):
        documents = self.retriever.retrieve(question)
        answer = self.generator.generate(question, documents)
        return answer
