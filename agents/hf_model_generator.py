# agents/hf_model_generator.py

import requests
import json

#API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
import os
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class HFHistBERTGenerator:
    def __init__(self):
        self.api_url = API_URL

    def generate(self, question, documents):
        context = ""
        for doc in documents[:3]:
            meta = doc.get("meta", {})
            chunk = doc.get("text", "").replace("\n", " ").strip()
            author = meta.get("author", "Unknown")
            date = meta.get("date", "Unknown")
            location = meta.get("location", "Unknown")
            context += f"{author}, {date}, {location}:\n{chunk}\n\n"

        prompt = f"Answer the question based on the following WW1 letters or diaries.\n\nContext:\n{context}\n\nQuestion: {question}"

        response = requests.post(self.api_url, headers=headers, json={"inputs": prompt})
        if response.status_code != 200:
            return f" API Error {response.status_code}: {response.text}"

        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "error" in result:
            return f" Hugging Face Error: {result['error']}"
        return " Unexpected response from Hugging Face API."
