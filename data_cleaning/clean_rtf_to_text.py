import os
from pathlib import Path
from striprtf.striprtf import rtf_to_text
import re

RAW_DATA_DIR = "data"
CLEANED_DATA_DIR = "data_cleaned"

def extract_rtf_text(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_rtf = f.read()
            return rtf_to_text(raw_rtf)
    except Exception as e:
        print(f"[ERROR] Failed to parse {file_path.name}: {e}")
        return None

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII
    text = re.sub(r'\s+', ' ', text)           # normalize whitespace
    text = re.sub(r'(Text:)+', 'Text:', text, flags=re.IGNORECASE)
    return text.strip()

def process_directory(subfolder):
    input_dir = Path(RAW_DATA_DIR) / subfolder
    output_dir = Path(CLEANED_DATA_DIR) / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    files_processed = 0
    for file in input_dir.glob("**/*.rtf"):
        plain_text = extract_rtf_text(file)
        if plain_text:
            cleaned = clean_text(plain_text)
            output_file = output_dir / file.with_suffix(".txt").name
            with open(output_file, "w", encoding="utf-8") as out:
                out.write(cleaned)
            files_processed += 1

    print(f"[✅] Processed {files_processed} RTF files in {subfolder}")

def main():
    for subfolder in ["Letters", "Diaries", "Others"]:
        process_directory(subfolder)

if __name__ == "__main__":
    main()
