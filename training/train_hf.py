from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json
import os

MODEL_NAME = "bert-base-uncased"  # You can change this to histbert-uncased or any base model
DATA_PATH = "training/processed/dataset.jsonl"
OUTPUT_DIR = "models/hf_model"

# 1. Load and prepare dataset
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]

raw_data = load_jsonl(DATA_PATH)
hf_dataset = Dataset.from_list([
    {"text": d["text"], "label": 1}  # Dummy label, for classification-style fine-tuning
    for d in raw_data
])

# 2. Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = hf_dataset.map(tokenize)

# 3. Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 4. Training setup
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    evaluation_strategy="no",
    logging_steps=10,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset
)

# 5. Train
trainer.train()

# 6. Save model and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"[✅] Fine-tuned model saved to {OUTPUT_DIR}")
