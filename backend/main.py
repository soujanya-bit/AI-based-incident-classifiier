from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from model import load_model, predict_incident, save_feedback
import uvicorn
import os

app = FastAPI(title="Adaptive Incident Classifier")

# load model at startup
MODEL = load_model(model_dir=os.getenv("MODEL_DIR", "./model"))

class IncidentIn(BaseModel):
    text: str
    metadata: Optional[dict] = None

class PredictionOut(BaseModel):
    category: str
    priority: str
    confidence: float

class FeedbackIn(BaseModel):
    text: str
    true_category: str
    true_priority: str
    metadata: Optional[dict] = None

@app.on_event("startup")
def startup_event():
    # ensure model loaded
    global MODEL
    MODEL = MODEL  # noop to ensure loaded

@app.post("/predict", response_model=PredictionOut)
def predict(inc: IncidentIn):
    try:
        category, priority, conf = predict_incident(inc.text, MODEL)
        return PredictionOut(category=category, priority=priority, confidence=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback(feedback: FeedbackIn):
    """
    Save user-corrected label for later retraining.
    This app appends to a CSV in ./feedback/feedback.csv by default.
    """
    try:
        save_feedback(feedback.dict(), path=os.getenv("FEEDBACK_PATH", "./feedback/feedback.csv"))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

# for local dev
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
