import os
import csv

DATA_DIR = "data"
OUTPUT_CSV = "data/training_generator.csv"

def extract(text, field):
    for line in text.splitlines():
        if line.strip().startswith(field):
            return line.split(":", 1)[1].strip()
    return ""

def extract_multiline(text, section):
    lines = text.splitlines()
    result = []
    collecting = False
    for line in lines:
        if line.startswith(section):
            collecting = True
            result.append(line.split(":", 1)[1].strip())
            continue
        if collecting and any(line.startswith(x) for x in ["Author:", "Date:", "Location:", "Recipient:", "Summary:", "Sentiment"]):
            break
        if collecting:
            result.append(line.strip())
    return " ".join(result)

def collect_examples():
    rows = []
    for folder in ["letters", "diaries"]:
        base_path = os.path.join(DATA_DIR, folder)
        for root, _, files in os.walk(base_path):
            for file in files:
                if not file.endswith(".txt"):
                    continue
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                author = extract(content, "Author:")
                date = extract(content, "Date:")
                location = extract(content, "Location:")
                text = extract_multiline(content, "Text:")

                if not text or len(text.split()) < 20:
                    continue  # skip short or empty texts

                context = f"Letter from {author} on {date} at {location}. {text}"
                question = f"What did {author} say?"
                answer = text.strip()

                rows.append((context, question, answer))
    return rows

if __name__ == "__main__":
    examples = collect_examples()
    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["context", "question", "answer"])
        writer.writerows(examples)

    print(f"[✅] Extracted {len(examples)} training examples to {OUTPUT_CSV}")
