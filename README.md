# 🔍 Hybrid Fake News Detection using Semantic Embeddings, Stylometric Features & Explainable AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)
![SBERT](https://img.shields.io/badge/SBERT-all--MiniLM--L6--v2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainable--AI-purple)

**A production-ready NLP project for detecting fake news using a hybrid approach combining semantic embeddings, stylometric analysis, and explainable AI.**

[Live Demo](#deployment) · [Architecture](#system-architecture) · [Results](#results) · [Installation](#installation)

</div>

---

## 📋 Project Overview

This project builds a comprehensive **Fake News Detection** system that goes beyond simple bag-of-words approaches. By combining:

- **Sentence-BERT** (384-dimensional semantic embeddings)
- **Stylometric features** (writing style, sentiment, readability)
- **Machine Learning classifiers** (Logistic Regression, XGBoost)
- **SHAP Explainability** (transparent AI decisions)
- **Cross-domain evaluation** (generalization testing)

The result is a hybrid model that achieves high accuracy while remaining interpretable and generalizable.

---

## ❓ Problem Statement

Fake news spreads faster than real news on social media and can have severe societal impacts — influencing elections, public health decisions, and social cohesion. Automated detection systems are needed to:

1. Scale beyond human fact-checking capacity
2. Provide real-time analysis
3. Explain *why* an article is classified as fake
4. Generalize across different news domains

Traditional approaches (TF-IDF + simple classifiers) overfit to training data. Our hybrid approach combines semantic understanding with stylometric analysis to detect deceptive writing patterns.

---

## 📦 Dataset Description

### Training & Testing: ISOT Fake News Dataset
| Property | Value |
|---|---|
| Source | University of Victoria |
| Files | `Fake.csv`, `True.csv` |
| Fake articles | ~23,500 |
| Real articles | ~21,400 |
| Topics | Politics, world news |
| Time period | 2016–2017 |

### Cross-Domain Evaluation: McIntire Dataset
| Property | Value |
|---|---|
| Source | Kaggle (George McIntire) |
| File | `fake_or_real_news.csv` |
| Articles | ~6,335 |
| Labels | FAKE / REAL |
| Purpose | Generalization testing ONLY (no training) |

---

## 🏗️ System Architecture

```
News Article
     │
     ▼
┌─────────────────────────────────┐
│         Text Cleaning            │
│  URL removal · HTML removal     │
│  Special chars · Normalization  │
└─────────────────────────────────┘
     │
     ├─────────────────────────────────────────┐
     ▼                                         ▼
┌──────────────┐                    ┌──────────────────────┐
│ SBERT Encoder│                    │ Stylometric Extractor│
│ all-MiniLM   │                    │                      │
│ L6-v2        │                    │ · Sentiment          │
│              │                    │ · Readability        │
│ 384-dim      │                    │ · Exclamation ratio  │
│ embeddings   │                    │ · Capital ratio      │
└──────┬───────┘                    │ · Sentence length    │
       │                            │ · Word/char count    │
       │                            └──────────┬───────────┘
       │                                       │
       └───────────────┬───────────────────────┘
                       ▼
              ┌─────────────────┐
              │  Feature Fusion  │
              │   392 features   │
              │  (384 + 8)       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   ML Classifier  │
              │  XGBoost / LR   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ SHAP Explainer  │
              │ Feature Impact  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Prediction    │
              │  FAKE / REAL    │
              │  + Confidence   │
              └─────────────────┘
```

---

## 🔄 Data Pipeline

```
Fake.csv + True.csv
        │
        ▼ Label encoding (Fake=1, Real=0)
        ▼ Merge & shuffle (seed=42)
        ▼ EDA & visualization
        │
        ▼ Text cleaning pipeline
        ▼ SBERT embeddings generation
        ▼ Stylometric feature extraction
        ▼ Feature fusion → features.npz
        │
        ▼ Train-test split (80/20, stratified)
        ▼ Model training (4 models)
        ▼ Evaluation & comparison
        ▼ Best model selection (F1)
        ▼ Model saved → models/
```

---

## ⚙️ Feature Engineering

### A. Semantic Features (SBERT)
| Model | Dimension | Normalization |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | L2 normalized |

### B. Stylometric Features
| Feature | Description | Range |
|---|---|---|
| `sentiment_polarity` | TextBlob sentiment | [-1, +1] |
| `sentiment_subjectivity` | Objective vs subjective | [0, 1] |
| `readability_score` | Flesch Reading Ease | [0, 100] |
| `exclamation_ratio` | `!` count / text length | [0, 1] |
| `capital_ratio` | UPPERCASE / total letters | [0, 1] |
| `avg_sentence_length` | Avg words per sentence | float |
| `word_count` | Total words | int |
| `char_count` | Total chars (no spaces) | int |

### C. Feature Fusion
```
X_hybrid = [SBERT (384) | Stylometric (8)] = 392 features
```

---

## 🤖 Model Comparison

| Model | Features | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| TF-IDF + LR | TF-IDF | ~95% | ~0.95 | ~0.98 |
| SBERT + LR | SBERT (384) | ~96% | ~0.96 | ~0.99 |
| Hybrid + LR | SBERT + Style | ~97% | ~0.97 | ~0.99 |
| **Hybrid + XGBoost** | **SBERT + Style** | **~98%** | **~0.98** | **~0.99** |

*Results are approximate and depend on the training run.*

---

## 🔬 Explainable AI (SHAP)

We use SHAP (SHapley Additive exPlanations) to explain model predictions:

- **LinearExplainer** → for Logistic Regression (fast, exact)
- **TreeExplainer** → for XGBoost (fast, exact)

SHAP outputs:
1. **Summary Plot**: Distribution of feature impacts across all test samples
2. **Bar Plot**: Mean absolute SHAP value (global feature importance)
3. **Single-instance explanations**: Which features pushed toward Fake/Real for any article

---

## 🌍 Cross-Domain Evaluation

The model is trained **only on ISOT** and evaluated on **McIntire** to measure generalization:

| Metric | ISOT Test | McIntire (Cross-Domain) | Drop % |
|---|---|---|---|
| Accuracy | ~98% | ~85-92% | ~6-13% |
| F1 Score | ~0.98 | ~0.85-0.92 | ~6-13% |

**Interpretation**: Some performance drop is expected due to:
- Different news sources and writing styles
- Different time periods
- Different distribution of fake vs real news topics

---

## 🖥️ Streamlit Application

### Features
- 📰 **Article Input**: Paste full article text
- 🚨 **Verdict**: FAKE / REAL with confidence score
- 📊 **Probability Distribution**: Bar chart of fake/real probabilities
- 🎭 **Sentiment Analysis**: Polarity & subjectivity scores
- 📐 **Article Statistics**: Word, character, sentence counts
- 🔬 **Stylometric Table**: All 8 stylometric features with descriptions
- 🧠 **SHAP Explanation**: Top features pushing toward each verdict

### Run Locally
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Results

### Training Results (ISOT Dataset)
- Best Model: **Hybrid + XGBoost**
- F1 Score: **~0.98**
- ROC-AUC: **~0.99**

### Key Findings
1. SBERT embeddings significantly outperform TF-IDF for cross-domain generalization
2. Stylometric features (especially exclamation ratio, capital ratio) are strong fake news indicators
3. Hybrid approach outperforms either feature type alone
4. XGBoost + hybrid features gives the best trade-off between accuracy and interpretability

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/Jaivardhan0307/FakeNewsNLP.git
cd FakeNewsNLP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Datasets

Place these files in the `data/` directory:

1. **ISOT Dataset** (Fake.csv + True.csv):
   → https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

2. **McIntire Dataset** (fake_or_real_news.csv):
   → https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news

---

## 🚀 Usage

### Step 1: Run EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Step 2: Generate Features
```bash
jupyter notebook notebooks/02_features.ipynb
```

### Step 3: Train Models
```bash
jupyter notebook notebooks/03_train.ipynb
```

### Step 4: Explainability
```bash
jupyter notebook notebooks/04_explainability.ipynb
```

### Step 5: Cross-Domain Evaluation
```bash
jupyter notebook notebooks/05_cross_domain_test.ipynb
```

### Step 6: Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## 🚢 Deployment

### 1. Streamlit Community Cloud (Free)
```bash
# Push to GitHub, then:
# Go to: https://share.streamlit.io
# Connect your GitHub repo
# Set main file: app/streamlit_app.py
```

### 2. Render
```bash
# Create new Web Service on render.com
# Connect GitHub repo
# Build command: pip install -r requirements.txt
# Start command: streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

### 3. Docker
```bash
# Build image
docker build -t fake-news-detector .

# Run container
docker run -p 8501:8501 fake-news-detector

# Open: http://localhost:8501
```

### 4. Hugging Face Spaces
```bash
# Create new Space on huggingface.co/spaces
# Select Streamlit SDK
# Upload all project files
# Set app_file: app/streamlit_app.py
```

> **Note**: For deployment, make sure to commit your trained model files (`models/*.pkl`) and outputs (`outputs/*.npz`, `outputs/*.pkl`). These are excluded from `.gitignore` by default.

---

## 🔭 Future Work

1. **Transformer Fine-tuning**: Fine-tune BERT/RoBERTa end-to-end on ISOT
2. **Multi-modal**: Include image analysis for articles with embedded images
3. **Knowledge Graph**: Fact-check against Wikidata
4. **Real-time API**: FastAPI backend for production deployment
5. **Active Learning**: Human-in-the-loop correction mechanism
6. **Multilingual**: Extend to non-English fake news datasets
7. **Temporal Analysis**: Track how fake news evolves over time

---

## 📁 Project Structure

```
FakeNewsNLP/
│
├── data/                    # Raw datasets (download required)
│   ├── Fake.csv
│   ├── True.csv
│   └── fake_or_real_news.csv
│
├── notebooks/               # Jupyter notebooks (run in order)
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_features.ipynb    # Feature Engineering
│   ├── 03_train.ipynb       # Model Training
│   ├── 04_explainability.ipynb  # SHAP Analysis
│   └── 05_cross_domain_test.ipynb  # Cross-domain Evaluation
│
├── src/                     # Modular Python source code
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loading utilities
│   ├── preprocessing.py     # Text cleaning pipeline
│   ├── feature_engineering.py  # SBERT + stylometric features
│   ├── model_utils.py       # Training & evaluation utilities
│   ├── explainability.py    # SHAP explanation utilities
│   └── predict.py           # Inference pipeline
│
├── app/                     # Streamlit web application
│   └── streamlit_app.py
│
├── outputs/                 # Generated outputs (auto-created)
│   ├── combined_news.csv
│   ├── features.npz
│   ├── stylometric_features.csv
│   ├── model_comparison.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_summary.png
│   └── shap_bar.png
│
├── models/                  # Saved models (auto-created)
│   ├── fake_news_model.pkl
│   └── model_metadata.pkl
│
├── .streamlit/              # Streamlit configuration
│   └── config.toml
│
├── README.md
├── requirements.txt
├── Dockerfile
├── Procfile
├── runtime.txt
├── .gitignore
└── LICENSE
```

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **ISOT Fake News Dataset**: University of Victoria Information Security and Object Technology Research Lab
- **McIntire Dataset**: George McIntire (Kaggle)
- **Sentence-BERT**: Reimers & Gurevych (2019)
- **SHAP**: Lundberg & Lee (2017)
- **XGBoost**: Chen & Guestrin (2016)

---

<div align="center">
Made with ❤️ for NLP Research · GitHub: <a href="https://github.com/Jaivardhan0307">Jaivardhan0307</a>
</div>
