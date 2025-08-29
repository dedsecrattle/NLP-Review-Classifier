import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from src.dataset import ReviewDataset
from src.model import DebertaClassifier

# Load dataset
df = pd.read_csv("data/reviews.csv")  # columns: text, label
texts = df["text"].tolist()
labels = df["label"].tolist()

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

# Datasets
train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)

# Model
model = DebertaClassifier()

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    torch.save(model.state_dict(), "./saved_model/model.pth")
