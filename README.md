# 📝 MBO-DeBERTa Review Classifier

A Machine Learning/NLP pipeline to automatically detect low-quality reviews (Advertisements, Irrelevant Content, Rants Without Visit) and classify Valid Reviews, optimized with Monarch Butterfly Optimizer (MBO) on top of DeBERTa.
Outputs confidence percentages per class, with a Streamlit demo app for real-time testing.

---

## 🚀 Features

• DeBERTa Transformer Backbone for powerful text encoding  
• MBO (Monarch Butterfly Optimization) for hyperparameter tuning  
• Softmax Confidence Scores for each predicted category  
• Hybrid Pipeline: rule-based + ML classification  
• Explainability (via SHAP) to highlight key words influencing predictions  
• Streamlit UI to demo review classification live

---

## 📂 Project Structure

```
mbo-deberta-project/
├── data/                    # Place your datasets here
│   └── reviews.csv          # Example CSV: text, label
├── src/                     # Core source files
│   ├── preprocess.py        # Text cleaning & preprocessing
│   ├── dataset.py           # Torch dataset wrapper
│   ├── model.py             # DeBERTa classifier with softmax
│   ├── train.py             # Training loop
│   ├── optimize.py          # MBO hyperparameter tuning
│   ├── inference.py         # Prediction with confidence %
│   └── utils.py             # Helper functions
├── app.py                   # Streamlit demo UI
├── pyproject.toml           # Dependencies (managed by uv)
├── uv.lock                  # Pinned versions
└── README.md                # You are here 🚀
```

---

## 🛠️ Setup Instructions

### 1️⃣ Clone Repo

```bash
git clone https://github.com/your-username/mbo-deberta-project.git
cd mbo-deberta-project
```

### 2️⃣ Sync Dependencies (using uv)

```bash
uv sync
```

### 3️⃣ Ensure pip is available inside .venv

```bash
uv venv --with-pip .venv
```

### 4️⃣ Install spaCy English model

Required for preprocessing (lemmatization, tokenization).

```bash
uv run python -m spacy download en_core_web_sm
```

### 5️⃣ (Optional) Add Dataset

Place your dataset in `data/reviews.csv` with format:

```csv
text,label
"Best pizza! Visit www.promo.com",Advertisement
"I love my new phone, but this place is noisy",Irrelevant
"Never been here but I heard it's bad",RantWithoutVisit
"Amazing food and friendly staff!",Valid
```

---

## ▶️ Usage

### 🔹 Train Model

```bash
uv run python -m src.train
```

### 🔹 Optimize Hyperparameters with MBO

```bash
uv run python -m src.optimize
```

### 🔹 Run Inference

```bash
uv run python -m src.inference
```

Example:

```python
from src.inference import classify_review
print(classify_review("Best pizza! Visit www.promo.com"))
```

Output:

```json
{
  "Predicted": "Advertisement",
  "Confidence": {
    "Advertisement": 92.5,
    "Irrelevant": 3.1,
    "RantWithoutVisit": 2.0,
    "Valid": 2.4
  }
}
```

### 🔹 Launch Streamlit Demo

```bash
uv run streamlit run app.py
```

---

## 📊 Evaluation

• Metrics: Precision, Recall, F1 per category  
• Visuals: Confusion Matrix, Confidence Distribution  
• Explainability: SHAP word highlights for interpretability

---

## 📚 Data Sources

• Google Maps Restaurant Reviews (Kaggle)  
• Google Local Reviews (UCSD)  
• Yelp Review Dataset  
• Amazon Polarity Reviews

---

## ✨ Future Improvements

• Add multi-lingual support  
• Deploy API with FastAPI  
• Extend rule-based filters for more ad/spam patterns  
• Ensemble with BiLSTM+Word2Vec baseline for robustness

---

## 👨‍💻 Authors / Team

• Prabhat & Anvita
