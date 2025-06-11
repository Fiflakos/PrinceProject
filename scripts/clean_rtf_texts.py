import os
from pathlib import Path
import re
from striprtf.striprtf import rtf_to_text

SOURCE_FOLDERS = ["data/Letters", "data/Diaries", "data/Others"]
DESTINATION_FOLDER = "data_cleaned"

def clean_text(text):
    text = re.sub(r"\\['a-zA-Z0-9]+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def convert_and_clean_rtf_file(source_path, target_path):
    try:
        with open(source_path, "r", encoding="utf-8", errors="ignore") as file:
            raw_rtf = file.read()
        plain_text = rtf_to_text(raw_rtf)
        cleaned = clean_text(plain_text)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as out_file:
            out_file.write(cleaned)
        print(f"[✅] Cleaned and saved: {target_path}")
    except Exception as e:
        print(f"[ERROR] Failed to parse {source_path}: {e}")

def main():
    for folder in SOURCE_FOLDERS:
        for rtf_file in Path(folder).rglob("*.rtf"):
            relative = rtf_file.relative_to("data")
            target_txt = Path(DESTINATION_FOLDER) / relative
            target_txt = target_txt.with_suffix(".txt")
            convert_and_clean_rtf_file(rtf_file, target_txt)

if __name__ == "__main__":
    main()
