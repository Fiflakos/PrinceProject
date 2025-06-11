# retrieval_modules/utils.py

import re
from typing import List, Dict
from nltk.tokenize import word_tokenize

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9.,;!?\'"()\-\n ]', '', text)
    return text.strip()

def tokenize(text: str) -> List[str]:
    return word_tokenize(text)

def generate_doc_id(path: str, prefix: str = "") -> str:
    base = path.split("/")[-1].replace(".txt", "").replace(".rtf", "")
    return f"{prefix}{base}"

def load_metadata_from_text(text: str) -> Dict[str, str]:
    metadata = {}
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            metadata[key.strip().lower()] = val.strip()
        else:
            break
    return metadata
