"""
Simple CLI tool to test the local fine-tuned Hugging Face model directly.
"""

from agents.hf_model_generator import LocalHistBERTGenerator

def main():
    model = LocalHistBERTGenerator()
    print("🤖 WW1 Assistant (Local HistBERT)")
    while True:
        query = input("🧑 You: ")
        if query.lower() in {"exit", "quit"}:
            break
        docs = [{
            "text": "We received letters from Melbourne. It was a great comfort for all of us. We fought hard and now have a rest. Lice are our biggest enemy, but we got sulphur powder and it helps.",
            "meta": {
                "author": "Norman Griffiths Ellsworth",
                "date": "14 August 1915",
                "location": "Gallipoli, Turkey"
            }
        }]
        answer = model.generate(query, docs)
        print("🤖 HistBERT:", answer, "
")

if __name__ == "__main__":
    main()
