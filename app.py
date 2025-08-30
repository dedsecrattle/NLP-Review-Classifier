import re
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, TrainerCallback
from torchvision import models, transforms
from PIL import Image

# ───────────────────────────────
# Preprocessing
# ───────────────────────────────
def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"\d+", "<NUM>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ───────────────────────────────
# Config
# ───────────────────────────────
MODEL_NAME = "microsoft/deberta-v3-base"
SAVE_DIR = "saved_model"
CATEGORIES = ["truthful", "deceptive"]
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}

st.set_page_config(page_title="Fake Review Classifier", layout="wide")

# ───────────────────────────────
# Image Feature Extraction
# ───────────────────────────────
def get_resnet18_encoder():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, tfm

def read_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def image_feat_histogram(img, bins_per_channel=8):
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        arr = np.stack([arr] * 3, axis=-1)
    hist = []
    for ch in range(3):
        h, _ = np.histogram(
            arr[:, :, ch], bins=bins_per_channel, range=(0, 255), density=True
        )
        hist.append(h)
    return np.concatenate(hist).astype("float32")

def extract_image_features(paths, method="ResNet18", base_dir=None):
    feats = []
    if method == "None":
        return np.zeros((len(paths), 1), dtype="float32")
    if method == "ColorHistogram":
        for p in paths:
            full = os.path.join(base_dir, p) if base_dir and not os.path.isabs(p) else p
            img = read_image(full)
            feats.append(image_feat_histogram(img) if img is not None else np.zeros(24))
        return np.vstack(feats)
    if method == "ResNet18":
        model, tfm = get_resnet18_encoder()
        for p in paths:
            full = os.path.join(base_dir, p) if base_dir and not os.path.isabs(p) else p
            img = read_image(full)
            if img is None:
                feats.append(np.zeros(512))
            else:
                with torch.no_grad():
                    x = tfm(img).unsqueeze(0)
                    feat = model(x).squeeze(0).numpy()
                feats.append(feat)
        return np.vstack(feats)
    return np.zeros((len(paths), 1), dtype="float32")

# ───────────────────────────────
# Model
# ───────────────────────────────
class DebertaWithImage(nn.Module):
    def __init__(self, model_name, num_labels=2, use_images=False, img_dim=512):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name)
        self.use_images = use_images
        text_dim = self.text_model.config.hidden_size
        fused_dim = text_dim + (img_dim if use_images else 0)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask, img_feats=None, labels=None):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        if self.use_images and img_feats is not None:
            fused = torch.cat([pooled, img_feats], dim=1)
        else:
            fused = pooled
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

# ───────────────────────────────
# Dataset
# ───────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, imgs=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.imgs = imgs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        if self.imgs is not None:
            item["img_feats"] = torch.tensor(self.imgs[idx], dtype=torch.float32)
        return item

# ───────────────────────────────
# Streamlit Trainer Callback
# ───────────────────────────────
class StreamlitCallback(TrainerCallback):
    def __init__(self, num_epochs):
        self.epoch_bar = st.progress(0)
        self.status_placeholder = st.empty()
        self.table_placeholder = st.empty()
        self.num_epochs = num_epochs
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.status_placeholder.text(
                f"Step {state.global_step} | Loss: {logs['loss']:.4f}"
            )
    def on_epoch_end(self, args, state, control, **kwargs):
        progress = (state.epoch or 0) / self.num_epochs
        self.epoch_bar.progress(progress)

# ───────────────────────────────
# Training
# ───────────────────────────────
def train_model(
    df, text_col, label_col, photo_col,
    use_images, image_method, base_dir,
    num_epochs=3, batch_size=8, lr=2e-5
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    texts = df[text_col].astype(str).apply(preprocess_text).tolist()
    labels = df[label_col].map(LABEL2ID).tolist()

    img_feats = None
    img_dim = 0
    if use_images and photo_col in df.columns:
        paths = df[photo_col].astype(str).tolist()
        img_feats = extract_image_features(paths, method=image_method, base_dir=base_dir)
        img_dim = img_feats.shape[1]

    # Train/val split
    train_texts, val_texts, train_labels, val_labels, train_imgs, val_imgs = train_test_split(
        texts, labels, img_feats if img_feats is not None else [None]*len(texts),
        test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = ReviewDataset(train_texts, train_labels, tokenizer,
                             train_imgs if img_feats is not None else None)
    val_ds   = ReviewDataset(val_texts, val_labels, tokenizer,
                             val_imgs if img_feats is not None else None)

    model = DebertaWithImage(MODEL_NAME, num_labels=len(CATEGORIES),
                             use_images=use_images, img_dim=img_dim)

    training_args = TrainingArguments(
        output_dir="./hf_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        report = classification_report(labels, preds, target_names=CATEGORIES, output_dict=True)
        return {
            "accuracy": report["accuracy"],
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[StreamlitCallback(num_epochs)]
    )

    with st.spinner("⏳ Training in progress..."):
        trainer.train()

    # Save best model (our format)
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model_weights.pt"))
    tokenizer.save_pretrained(SAVE_DIR)
    with open(os.path.join(SAVE_DIR, "image_config.json"), "w") as f:
        json.dump({"use_images": use_images, "image_method": image_method, "img_dim": img_dim}, f)

    # Final evaluation report
    return trainer.evaluate()

# ───────────────────────────────
# Inference
# ───────────────────────────────
def load_model_for_inference():
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, use_fast=False)
    cfg = json.load(open(os.path.join(SAVE_DIR, "image_config.json")))
    model = DebertaWithImage(MODEL_NAME, num_labels=len(CATEGORIES),
                             use_images=cfg["use_images"], img_dim=cfg["img_dim"])
    state = torch.load(os.path.join(SAVE_DIR, "model_weights.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return tokenizer, model, cfg

def predict_texts(texts, photo_paths=None, base_dir=None):
    texts = [preprocess_text(t) for t in texts]
    tokenizer, model, cfg = load_model_for_inference()
    enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    img_feats = None
    if cfg["use_images"] and photo_paths:
        img_feats = extract_image_features(photo_paths, method=cfg["image_method"], base_dir=base_dir)
        img_feats = torch.tensor(img_feats, dtype=torch.float32)
    with torch.no_grad():
        data = model(enc["input_ids"], enc["attention_mask"], img_feats)
        logits = data["logits"]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return [ID2LABEL[p] for p in preds], probs

# ───────────────────────────────
# Streamlit UI
# ───────────────────────────────
st.title("🕵️ Truthful vs Deceptive Review Classifier (DeBERTa + Optional Images)")

with st.sidebar:
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    col_text = st.text_input("Review column", "review")
    col_label = st.text_input("Label column", "label")
    col_photo = st.text_input("Photo column (optional)", "photo")

    num_epochs = st.slider("Epochs", 1, 5, 3)
    batch_size = st.slider("Batch size", 4, 32, 8, step=4)
    lr = st.selectbox("Learning rate", [5e-5, 3e-5, 2e-5, 1e-5, 1e-6], index=2)

    use_images = st.checkbox("Use images", value=False)
    image_method = st.selectbox("Image features", ["ResNet18", "ColorHistogram", "None"], index=0)
    base_dir = st.text_input("Image base dir (optional)", "")

    btn_train = st.button("Train Model")

if csv_file:
    df = pd.read_csv(csv_file)
    st.write("### Preview", df.head())
    if btn_train:
        df = df[df[col_label].isin(CATEGORIES)]
        if df.empty:
            st.error(f"Label column must contain: {CATEGORIES}")
        else:
            report = train_model(df, col_text, col_label, col_photo,
                                 use_images, image_method, base_dir,
                                 num_epochs=num_epochs, batch_size=batch_size, lr=lr)
            st.success("✅ Training complete. Model saved in saved_model/")
            st.write("### Evaluation Report (Validation Set, Best Model)")
            st.json(report)

# Inference
st.subheader("🔮 Quick Inference")
user_txt = st.text_area("Enter review text")
uploaded_img = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
tmp_path = None
if uploaded_img:
    tmp_path = "uploaded_tmp.png"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_img.read())

if st.button("Classify Review"):
    if not os.path.exists(SAVE_DIR):
        st.error("No trained model found. Train first.")
    elif not user_txt.strip():
        st.warning("Enter review text.")
    else:
        img_path = tmp_path if tmp_path else None
        labels, probs = predict_texts([user_txt],
                                      photo_paths=[img_path] if img_path else None,
                                      base_dir=base_dir)
        st.write("**Prediction:**", labels[0])
        st.json({CATEGORIES[i]: float(probs[0][i]) for i in range(len(CATEGORIES))})
