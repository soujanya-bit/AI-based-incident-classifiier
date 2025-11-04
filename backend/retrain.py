"""
retrain.py
Run this script to retrain the classifier using feedback data.
Writes fine-tuned model to ./model
"""

import os
import pandas as pd
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

BASE_MODEL = "distilbert-base-uncased"
MODEL_OUT = "./model"
FEEDBACK_PATH = "./feedback/feedback.csv"
NUM_LABELS = 5  # update according to your labels

LABEL_MAP = {
    "Network Issue": 0,
    "Access Request": 1,
    "Software": 2,
    "Infrastructure": 3,
    "Other": 4
}

def load_feedback(path=FEEDBACK_PATH):
    if not os.path.exists(path):
        raise RuntimeError("No feedback data found at " + path)
    df = pd.read_csv(path)
    # ensure columns exist
    df = df.dropna(subset=["text", "true_category"])
    df["label"] = df["true_category"].map(LABEL_MAP).fillna(LABEL_MAP["Other"]).astype(int)
    return df

def preprocess(tokenizer, texts):
    return tokenizer(texts, truncation=True, padding=True)

def main():
    df = load_feedback()
    # if you have an original training dataset, merge it here
    # For demo: use feedback only (not ideal)
    ds = Dataset.from_pandas(df[["text", "label"]])
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir="./tmp_trainer",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        remove_unused_columns=False
    )

    def compute_metrics(p):
        metric = load_metric("accuracy")
        preds = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # Save fine-tuned model
    os.makedirs(MODEL_OUT, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print("Saved fine-tuned model to", MODEL_OUT)

if __name__ == "__main__":
    main()
