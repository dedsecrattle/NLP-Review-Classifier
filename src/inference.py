import torch
from transformers import DebertaTokenizer
from src.model import DebertaClassifier

labels_map = {0: "Advertisement", 1: "Irrelevant", 2: "RantWithoutVisit", 3: "Valid"}

tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = DebertaClassifier()
model.load_state_dict(torch.load("saved_model/pytorch_model.bin"))
model.eval()


def classify_review(text: str):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        logits, probs = model(**inputs)
    probs = probs[0].tolist()
    confidences = {
        labels_map[i]: round(probs[i] * 100, 2) for i in range(len(labels_map))
    }
    predicted_class = labels_map[probs.index(max(probs))]
    return predicted_class, confidences
