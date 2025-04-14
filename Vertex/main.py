from fastapi import FastAPI, Request
from typing import List
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

# Load model and tokenizer
model_path = "saved_model"  # Adjust path if needed
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()

    # Expecting format: {"instances": [{"texts": ["text1", "text2", ...]}]}
    instances = body.get("instances", [])
    if not instances or "texts" not in instances[0]:
        return {"error": "Invalid input format. Expected {'instances': [{'texts': [ ... ]}]}"}

    texts: List[str] = instances[0]["texts"]

    # Tokenize input
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    results = [label_map[int(pred)] for pred in predictions]
    return {"predictions": results}
