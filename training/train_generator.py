import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

def load_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df[['context', 'question', 'answer']])

def preprocess(batch):
    # Prepare input strings
    inputs = [f"question: {q}  context: {c}" for q, c in zip(batch["question"], batch["context"])]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Tokenize targets/labels
    labels = tokenizer(
        batch["answer"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    model_name = "google/flan-t5-base"
    global tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load and preprocess data
    dataset = load_dataset_from_csv("data_cleaned/training_data.csv")
    tokenized_dataset = dataset.map(preprocess, batched=True)

    # Training configuration
    args = TrainingArguments(
        output_dir="./models/hf_generator",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=True,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset
    )

    # Train and save model
    trainer.train()
    trainer.save_model("./models/hf_generator")
    tokenizer.save_pretrained("./models/hf_generator")

if __name__ == "__main__":
    main()
