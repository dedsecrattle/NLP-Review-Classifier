import torch
import os
from transformers import AutoTokenizer
from src.model import DebertaClassifier

labels_map = {0: "Computer Generated", 1: "Human Generated"}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = DebertaClassifier()

# Load model weights with error handling
model_path = "./saved_model/model.pth"
if os.path.exists(model_path):
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
else:
    raise FileNotFoundError(
        f"Model file not found at {model_path}. Please train the model first."
    )

model.to(device)
model.eval()


def classify_review(text: str):
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    # Tokenize and convert to tensors
    inputs = tokenizer(
        text, truncation=True, padding=True, max_length=256, return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        outputs = model(**inputs)
        logits = outputs.logits
        probs = model.get_probabilities(logits)

    # Convert to list and get predictions
    probs = probs[0].cpu().tolist()
    confidences = {
        labels_map[i]: round(probs[i] * 100, 2) for i in range(len(labels_map))
    }

    # Get predicted class using argmax on logits
    predicted_idx = torch.argmax(logits, dim=-1).item()
    predicted_class = labels_map[predicted_idx]

    return predicted_class, confidences
