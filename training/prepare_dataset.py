import os
import json
import re
from scripts.chunk_utils import split_into_chunks
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_DIR = "training/processed"
MAX_CHARS = 400

def extract_metadata(text: str, field: str) -> str:
    for line in text.splitlines():
        if line.strip().lower().startswith(field.lower()):
            return line.split(":", 1)[1].strip()
    return ""

def extract_text_section(text: str) -> str:
    lines = text.splitlines()
    collecting = False
    result = []

    for line in lines:
        if line.strip().startswith("Text:"):
            collecting = True
            result.append(line.split(":", 1)[1].strip())
            continue
        if collecting:
            if any(line.strip().startswith(prefix) for prefix in [
                "Author:", "Date:", "Location:", "Summary:", "Entities:",
                "Keywords:", "Recipient:", "Sentiment"
            ]):
                break
            result.append(line.strip())

    return " ".join(result).strip()

def process_folder(folder_path: str, label: str, out_data: list):
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(".txt"):
                file_path = os.path.join(root, fname)
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                author = extract_metadata(content, "Author")
                date = extract_metadata(content, "Date")
                location = extract_metadata(content, "Location")
                full_text = extract_text_section(content)

                if not full_text:
                    print(f"[SKIPPED] No text found in {fname}")
                    continue

                chunks = split_into_chunks(full_text, max_chars=MAX_CHARS)
                for i, chunk in enumerate(chunks):
                    item = {
                        "id": f"{os.path.splitext(fname)[0]}_chunk{i+1}",
                        "type": label,
                        "author": author,
                        "date": date,
                        "location": location,
                        "source_file": fname,
                        "text": chunk
                    }
                    out_data.append(item)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_data = []

    print("[INFO] Processing letters...")
    process_folder(os.path.join(DATA_DIR, "letters"), "letter", out_data)

    print("[INFO] Processing diaries...")
    process_folder(os.path.join(DATA_DIR, "diaries"), "diary", out_data)

    out_file = os.path.join(OUTPUT_DIR, "dataset.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in tqdm(out_data):
            f.write(json.dumps(entry) + "\n")

    print(f"[DONE] Saved {len(out_data)} entries to {out_file}")

if __name__ == "__main__":
    main()
