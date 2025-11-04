import os
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import joblib

MODEL_NAME_FALLBACK = "distilbert-base-uncased"  # base model if no fine-tuned weights present
LABELS = ["Network Issue", "Access Request", "Software", "Infrastructure", "Other"]
PRIORITY = ["Low", "Medium", "High"]

class ModelWrapper:
    def __init__(self, model, tokenizer, labels=None):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels or LABELS

def load_model(model_dir: str = "./model") -> ModelWrapper:
    """
    If model_dir contains a HuggingFace model, load it; otherwise load base distilbert (zero-shot-ish).
    """
    # If a local fine-tuned model exists, load it
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            print(f"Loaded model from {model_dir}")
            return ModelWrapper(model=model, tokenizer=tokenizer)
        except Exception as e:
            print("Failed to load local model:", e)

    # Fallback to base model with small classifier head (not fine-tuned)
    print("Loading fallback base model (not fine-tuned). Predictions will be random-ish until fine-tuned.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FALLBACK)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_FALLBACK, num_labels=len(LABELS))
    return ModelWrapper(model=model, tokenizer=tokenizer)

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum(axis=-1, keepdims=True)

def predict_incident(text: str, model_wrapper: ModelWrapper) -> Tuple[str, str, float]:
    """
    Returns (category, priority, confidence).
    Currently priority is derived heuristically from confidence; extend later with a dedicated head.
    """
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    model.eval()

    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

    category = model_wrapper.labels[idx] if model_wrapper.labels else LABELS[idx]
    # simple priority mapping
    if confidence > 0.75:
        priority = "High"
    elif confidence > 0.5:
        priority = "Medium"
    else:
        priority = "Low"

    return category, priority, confidence

def save_feedback(feedback: dict, path: str = "./feedback/feedback.csv"):
    """
    Append user feedback to CSV. Required fields: text, true_category, true_priority.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {
        "text": feedback.get("text"),
        "true_category": feedback.get("true_category"),
        "true_priority": feedback.get("true_priority")
    }
    # write header if not exists
    if not os.path.exists(path):
        df = pd.DataFrame([row])
        df.to_csv(path, index=False)
    else:
        df = pd.DataFrame([row])
        df.to_csv(path, mode="a", header=False, index=False)
