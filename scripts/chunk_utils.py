# scripts/chunk_utils.py

import os
import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

def clean_text(text):
    """
    Removes unwanted characters and whitespace from the input text.
    """
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def load_and_chunk_documents(data_path, chunk_size=300):
    """
    Loads .txt documents from the data_cleaned directory and splits them into chunks.
    """
    chunks = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    text = clean_text(text)
                    sentences = sent_tokenize(text)

                    current_chunk = []
                    current_length = 0
                    for sentence in sentences:
                        current_chunk.append(sentence)
                        current_length += len(sentence)

                        if current_length >= chunk_size:
                            chunk = " ".join(current_chunk)
                            chunks.append({
                                "source": file,
                                "text": chunk
                            })
                            current_chunk = []
                            current_length = 0

                    # Add remaining chunk if any
                    if current_chunk:
                        chunk = " ".join(current_chunk)
                        chunks.append({
                            "source": file,
                            "text": chunk
                        })
    return chunks
