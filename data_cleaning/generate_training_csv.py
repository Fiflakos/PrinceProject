#!/usr/bin/env python3


import os
import csv
from pathlib import Path
import re
import random

CLEANED_DATA_DIR = "data_cleaned"
OUTPUT_CSV_PATH = Path(CLEANED_DATA_DIR) / "training_data.csv"

def extract_text_section(text):
    match = re.search(r'Text:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None

def generate_question_answer(text):
    # basic QA generator: split sentences and ask "What is this about?"
    sentences = re.split(r'\.|\n', text)
    context = ". ".join(sentences[:5]).strip()
    question = "What is this about?"
    answer = sentences[0].strip() if sentences else ""
    return context, question, answer

def collect_examples():
    examples = []
    for folder in ["Letters", "Diaries", "Others"]:
        folder_path = Path(CLEANED_DATA_DIR) / folder
        for file_path in folder_path.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                text = extract_text_section(content)
                if text:
                    context, question, answer = generate_question_answer(text)
                    if context and answer:
                        examples.append({"context": context, "question": question, "answer": answer})
            except Exception as e:
                print(f"[ERROR] {file_path.name}: {e}")
    return examples

def save_to_csv(examples):
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["context", "question", "answer"])
        writer.writeheader()
        for ex in examples:
            writer.writerow(ex)
    print(f"[✅] Saved {len(examples)} examples to {OUTPUT_CSV_PATH}")

def main():
    examples = collect_examples()
    random.shuffle(examples)
    save_to_csv(examples)

if __name__ == "__main__":
    main()
