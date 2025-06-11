import requests
from typing import List

import os
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

#API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

class APIHistBERTGenerator:
    def __init__(self):
        pass
    
    def generate(self, question: str, documents: List[dict]) -> str:
    
        context = ""
        for doc in documents[:3]:
            meta = doc.get("meta", {})
            chunk = doc.get("text", "").replace("\n", " ").strip()
            author = meta.get("author", "Unknown")
            date = meta.get("date", "Unknown")
            location = meta.get("location", "Unknown")
            context += f"{author}, {date}, {location}:\n{chunk}\n\n"

        prompt = f"Answer the question based on the following WW1 letters or diaries.\n\nContext:\n{context}\n\nQuestion: {question}"

        # Send the prompt to the Hugging Face Inference API
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})

        if response.status_code == 200:
            result = response.json()
            return result[0]["generated_text"]
        else:
            raise RuntimeError(f"API request failed: {response.status_code} {response.text}")
