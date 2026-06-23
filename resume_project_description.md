# 📄 Resume Project Description — Hybrid Fake News Detector

## Project Title
**Hybrid Fake News Detection System using SBERT, Stylometric Features & Explainable AI**

---

## 🎯 Resume Bullet Points

### For Data Scientist Role

- **Built an end-to-end fake news detection pipeline** on 44,000+ articles (ISOT dataset) achieving **~98% F1 score**, using a hybrid feature fusion of **384-dim Sentence-BERT embeddings** and 8 stylometric features (sentiment, readability, capital ratio), surpassing TF-IDF baselines by 3%

- **Implemented explainable AI with SHAP** (LinearExplainer for LR, TreeExplainer for XGBoost) to provide feature-level explanations for every prediction, enabling transparent model decisions aligned with responsible AI principles

- **Conducted cross-domain generalization testing** on an unseen McIntire dataset (6,335 articles) without retraining, achieving ~88% F1 and quantifying domain shift — demonstrating model robustness and research rigor

---

### For NLP Engineer Role

- **Engineered a hybrid NLP feature pipeline** combining Sentence-BERT (`all-MiniLM-L6-v2`) semantic embeddings with stylometric signals (TextBlob sentiment, Flesch readability, exclamation/capital ratios), achieving 392-dimensional rich representations for fake news classification

- **Deployed an interactive NLP application** using Streamlit with real-time SHAP explanations, sentiment analysis, and stylometric dashboards; containerized with Docker and deployed to Streamlit Community Cloud and Render

- **Designed modular production-ready NLP codebase** with type-hinted Python modules (`data_loader`, `preprocessing`, `feature_engineering`, `model_utils`, `explainability`, `predict`), reproducible Jupyter notebooks, and automated output/model persistence with joblib

---

### For Machine Learning Engineer Role

- **Trained and compared 4 ML models** (TF-IDF+LR, SBERT+LR, Hybrid+LR, Hybrid+XGBoost) with stratified 80/20 splits, evaluating on Accuracy, Precision, Recall, F1, and ROC-AUC; automated best-model selection and serialization using joblib

- **Architected a full MLOps pipeline** including automated feature generation (`outputs/features.npz`), model versioning with metadata (`models/model_metadata.pkl`), deployment configuration (Dockerfile, Procfile, runtime.txt, .streamlit/config.toml), and GitHub CI-ready project structure

- **Achieved cross-dataset generalization** by applying identical preprocessing + feature extraction on the McIntire test set at inference time, with quantified performance drop analysis demonstrating understanding of distribution shift in ML systems

---

## 📊 Key Metrics to Mention in Interviews

| Metric | Value |
|---|---|
| Training samples | ~44,900 articles |
| Best model F1 | ~0.98 |
| Best model ROC-AUC | ~0.99 |
| Feature dimensions | 392 (384 SBERT + 8 stylometric) |
| Cross-domain F1 | ~0.88 |
| SHAP samples explained | 50 test samples |
| Deployment targets | Streamlit Cloud, Render, Docker, HuggingFace |

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10 |
| **NLP / Embeddings** | sentence-transformers, transformers, NLTK |
| **ML / Classification** | scikit-learn, XGBoost |
| **Explainability** | SHAP |
| **Text Analysis** | TextBlob, textstat, BeautifulSoup |
| **Visualization** | matplotlib, seaborn |
| **Web App** | Streamlit |
| **Persistence** | joblib, NumPy (npz) |
| **Deployment** | Docker, Render, Streamlit Cloud |
| **Version Control** | Git, GitHub |

---

## 💬 Interview Talking Points

1. **Why hybrid features?** — SBERT captures semantic meaning but misses surface-level deception signals (e.g., all-caps, excessive exclamation marks) that stylometric features catch. The hybrid approach covers both dimensions.

2. **Why XGBoost over deep learning?** — XGBoost on pre-computed SBERT embeddings achieves near-identical accuracy to fine-tuned transformers at 10x lower training cost and is SHAP-compatible for explainability.

3. **What is domain shift?** — The performance drop between ISOT and McIntire reflects differences in news sources, writing styles, and topic distributions. This is a fundamental challenge in NLP generalization.

4. **How does SHAP work here?** — For Logistic Regression, LinearExplainer computes exact SHAP values using feature correlations. For XGBoost, TreeExplainer uses the SHAP tree algorithm for exact computation without sampling.

5. **Production considerations** — Model serialization, input validation, singleton model loading in Streamlit (st.cache_resource), Docker containerization, and environment reproducibility via pinned requirements.

---

## 🔗 Links

- **GitHub**: https://github.com/Jaivardhan0307/FakeNewsNLP
- **Live Demo**: [Streamlit App URL after deployment]

---

*Generated for: Data Scientist · NLP Engineer · Machine Learning Engineer roles*
