"""
Hybrid Fake News Detector — Streamlit Application
===================================================
A professional web application for detecting fake news articles
using SBERT embeddings, stylometric features, and SHAP explanations.

Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# NLTK data download (needed for TextBlob on cloud)
# ─────────────────────────────────────────────

@st.cache_resource
def download_nltk_data():
    """Download NLTK corpora required by TextBlob."""
    import nltk
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "wordnet", "stopwords", "brown"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

download_nltk_data()

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Hybrid Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Main Background ── */
  .stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
  }

  /* ── Hero Header ── */
  .hero-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6441a5 100%);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 20px 60px rgba(100, 65, 165, 0.4);
    border: 1px solid rgba(255,255,255,0.1);
  }

  .hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 20px rgba(0,0,0,0.5);
  }

  .hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.75);
    margin-top: 12px;
    font-weight: 300;
  }

  /* ── Result Cards ── */
  .result-fake {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(255, 65, 108, 0.5);
    animation: pulse-red 2s infinite;
    border: 2px solid rgba(255,150,150,0.3);
  }

  .result-real {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(56, 239, 125, 0.4);
    animation: pulse-green 2s infinite;
    border: 2px solid rgba(100,255,150,0.3);
  }

  @keyframes pulse-red {
    0%, 100% { box-shadow: 0 15px 40px rgba(255, 65, 108, 0.5); }
    50% { box-shadow: 0 20px 60px rgba(255, 65, 108, 0.8); }
  }

  @keyframes pulse-green {
    0%, 100% { box-shadow: 0 15px 40px rgba(56, 239, 125, 0.4); }
    50% { box-shadow: 0 20px 60px rgba(56, 239, 125, 0.7); }
  }

  .result-label {
    font-size: 3.5rem;
    font-weight: 700;
    color: white;
    margin: 0;
    text-shadow: 0 3px 15px rgba(0,0,0,0.4);
  }

  .result-emoji {
    font-size: 3rem;
    display: block;
    margin-bottom: 10px;
  }

  .confidence-text {
    font-size: 1.2rem;
    color: rgba(255,255,255,0.9);
    margin-top: 8px;
    font-weight: 500;
  }

  /* ── Metric Cards ── */
  .metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  .metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(100,65,165,0.3);
  }

  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
  }

  .metric-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
  }

  /* ── Section Cards ── */
  .section-card {
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
  }

  .section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e3f 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #6441a5, #2a5298);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 40px;
    font-size: 1.1rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 6px 20px rgba(100,65,165,0.4);
    width: 100%;
  }

  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(100,65,165,0.6);
  }

  /* ── Text Area ── */
  .stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    line-height: 1.6;
  }

  /* ── Progress bars ── */
  .stProgress > div > div > div {
    border-radius: 10px;
  }

  /* ── Hide streamlit branding ── */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load all models once and cache them."""
    try:
        from src.predict import load_all_models
        sbert, clf, meta = load_all_models(
            model_path=str(ROOT / "models" / "fake_news_model.pkl"),
            metadata_path=str(ROOT / "models" / "model_metadata.pkl"),
        )
        return sbert, clf, meta, None
    except FileNotFoundError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, f"Model loading error: {str(e)}"


@st.cache_resource
def load_background_data():
    """Load background data for SHAP."""
    features_path = ROOT / "outputs" / "features.npz"
    if features_path.exists():
        data = np.load(str(features_path), allow_pickle=True)
        return data["X"][:300]  # Use subset for speed
    return None


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def make_gauge_chart(probability: float, label: str) -> plt.Figure:
    """Create a minimal probability gauge."""
    fig, ax = plt.subplots(figsize=(4, 2.5), facecolor="none")

    fake_prob = probability
    real_prob = 1 - probability

    colors = ["#ff416c", "#38ef7d"]
    bars = ax.barh(
        ["Fake", "Real"],
        [fake_prob, real_prob],
        color=colors,
        height=0.5,
        edgecolor="none",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=9)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    for bar, val in zip(bars, [fake_prob, real_prob]):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left",
            color="white", fontsize=9, fontweight="bold"
        )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_tick_params(color="none")
    ax.set_title("Probability Distribution", color="white", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def render_sentiment_gauge(polarity: float, subjectivity: float):
    """Render sentiment metrics."""
    col1, col2 = st.columns(2)
    with col1:
        sentiment_label = "Positive" if polarity > 0.05 else ("Negative" if polarity < -0.05 else "Neutral")
        sentiment_color = "#38ef7d" if polarity > 0.05 else ("#ff416c" if polarity < -0.05 else "#a78bfa")
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{sentiment_color}">{polarity:+.3f}</div>
          <div class="metric-label">Polarity — {sentiment_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        subj_label = "Subjective" if subjectivity > 0.5 else "Objective"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#60a5fa">{subjectivity:.3f}</div>
          <div class="metric-label">Subjectivity — {subj_label}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔍 Fake News Detector")
    st.markdown("---")
    st.markdown("""
    **How it works:**

    1. 📝 Paste your article text
    2. 🧠 Click **Analyze News**
    3. 📊 View AI prediction & explanation

    ---

    **Model Architecture:**
    - 🤖 SBERT (Semantic Embeddings)
    - 📐 Stylometric Features
    - 🌳 XGBoost / Logistic Regression
    - 🔬 SHAP Explainability

    ---
    """)

    st.markdown("**📈 Features Used:**")
    features_list = [
        "384-dim SBERT embeddings",
        "Sentiment Polarity",
        "Sentiment Subjectivity",
        "Flesch Readability Score",
        "Exclamation Mark Ratio",
        "Capital Letter Ratio",
        "Avg. Sentence Length",
        "Word Count",
        "Character Count",
    ]
    for f in features_list:
        st.markdown(f"  • {f}")

    st.markdown("---")
    st.markdown("**⚠️ Disclaimer:**")
    st.caption("This tool is for research purposes. Always verify news with trusted sources.")


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-header">
  <p class="hero-title">🔍 Hybrid Fake News Detector</p>
  <p class="hero-subtitle">
    AI-powered detection using Sentence-BERT · Stylometric Analysis · SHAP Explainability
  </p>
</div>
""", unsafe_allow_html=True)

# Load models
sbert_model, classifier, metadata, model_error = load_models()

if model_error:
    st.warning(f"""
    ⚠️ **Model files not found in this deployment.**

    {model_error}

    The trained classifier (`models/fake_news_model.pkl`) needs to be included in the repository.
    The SBERT embedding model (`all-MiniLM-L6-v2`) will be downloaded automatically from HuggingFace on first run.
    """)

# Display model info
if metadata:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">✅</div>
          <div class="metric-label">Model Ready</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        model_name = metadata.get("model_name", "Hybrid Model")
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="font-size:1rem;">{model_name[:20]}</div>
          <div class="metric-label">Active Model</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        f1 = metadata.get("metrics", {}).get("f1", 0)
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{f1:.3f}</div>
          <div class="metric-label">Training F1 Score</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input Section ──────────────────────────────────────

st.markdown("### 📰 Paste Your Article")
article_text = st.text_area(
    label="Enter the news article text below:",
    placeholder="""Paste the full news article text here...

Example: "Scientists at MIT have discovered a revolutionary method to detect misinformation in news articles using advanced machine learning algorithms. The research, published in Nature, shows 98% accuracy..." """,
    height=220,
    label_visibility="collapsed",
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_clicked = st.button("🔍 Analyze News", key="analyze_btn")

# ── Analysis ────────────────────────────────────────────

if analyze_clicked:
    # Guard: check if model is loaded
    if model_error or sbert_model is None or classifier is None:
        st.error("❌ **Cannot analyze** — model files are not loaded. Please ensure `models/fake_news_model.pkl` and `models/model_metadata.pkl` are committed to the GitHub repository.")
        st.stop()

    # Input validation
    if not article_text or len(article_text.strip()) < 50:
        st.warning("⚠️ Please enter a news article with at least 50 characters for accurate analysis.")
        st.stop()

    with st.spinner("🧠 Analyzing article with AI models..."):
        try:
            from src.predict import predict_article, predict_with_shap
            from src.feature_engineering import get_feature_names

            # Load background for SHAP
            X_background = load_background_data()

            # Run prediction + SHAP
            result = predict_with_shap(
                article_text,
                X_train_sample=X_background,
                top_n=10,
                model_path=str(ROOT / "models" / "fake_news_model.pkl"),
                metadata_path=str(ROOT / "models" / "model_metadata.pkl"),
            )

        except Exception as e:
            st.error(f"""
            ❌ **Analysis Failed!**

            An error occurred during the prediction or explanation pipeline:

            **Error details:** `{e}`

            **How to resolve this:**
            1. Ensure all model files are present in the `models/` directory.
            2. Check that all dependencies in `requirements.txt` were installed correctly.
            3. Try refreshing the page — the SBERT model may still be downloading on first run.
            """)
            st.stop()

    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    # ── Verdict ────────────────────────────────────────
    prediction = result["prediction"]
    confidence = result["confidence"]

    col_verdict, col_probs = st.columns([1, 1])

    with col_verdict:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-fake">
              <span class="result-emoji">🚨</span>
              <p class="result-label">FAKE NEWS</p>
              <p class="confidence-text">Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-real">
              <span class="result-emoji">✅</span>
              <p class="result-label">REAL NEWS</p>
              <p class="confidence-text">Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_probs:
        st.markdown("#### Probability Distribution")
        fake_prob = result["fake_probability"]
        real_prob = result["real_probability"]

        st.markdown("**🚨 Fake News**")
        st.progress(fake_prob)
        st.caption(f"{fake_prob:.3%}")

        st.markdown("**✅ Real News**")
        st.progress(real_prob)
        st.caption(f"{real_prob:.3%}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Article Statistics ─────────────────────────────
    st.markdown("### 📐 Article Statistics")
    stat_cols = st.columns(4)

    stats_data = [
        ("💬", result["word_count"], "Words"),
        ("🔤", result["char_count"], "Characters"),
        ("📝", result["sentence_count"], "Sentences"),
        ("📖", f"{result['readability_score']:.1f}", "Readability"),
    ]

    for col, (icon, val, label) in zip(stat_cols, stats_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:2rem;">{icon}</div>
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sentiment Analysis ─────────────────────────────
    st.markdown("### 🎭 Sentiment Analysis")
    render_sentiment_gauge(
        result["sentiment_polarity"],
        result["sentiment_subjectivity"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stylometric Features Table ─────────────────────
    st.markdown("### 🔬 Stylometric Features")
    style_df = pd.DataFrame(
        list(result["stylometric_features"].items()),
        columns=["Feature", "Value"]
    )
    style_df["Value"] = style_df["Value"].apply(lambda x: f"{x:.4f}")
    feature_descriptions = {
        "sentiment_polarity": "Text sentiment direction [-1 to +1]",
        "sentiment_subjectivity": "Opinion vs fact [0=objective, 1=subjective]",
        "readability_score": "Flesch Reading Ease [0-100]",
        "exclamation_ratio": "Ratio of ! to total characters",
        "capital_ratio": "Ratio of UPPERCASE letters",
        "avg_sentence_length": "Average words per sentence",
        "word_count": "Total word count",
        "char_count": "Total character count (no spaces)",
    }
    style_df["Description"] = style_df["Feature"].map(feature_descriptions)
    st.dataframe(style_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP Explanation ──────────────────────────────
    if result.get("shap_explanation"):
        st.markdown("### 🔬 SHAP Feature Explanation")
        st.caption("Features pushing the prediction in each direction")

        shap_data = result["shap_explanation"]
        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown("**🚨 Toward FAKE (positive SHAP)**")
            if shap_data["top_positive"]:
                pos_df = pd.DataFrame(shap_data["top_positive"])
                pos_df["shap_value"] = pos_df["shap_value"].apply(lambda x: f"+{x:.4f}")
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
            else:
                st.info("No positive SHAP features")

        with col_neg:
            st.markdown("**✅ Toward REAL (negative SHAP)**")
            if shap_data["top_negative"]:
                neg_df = pd.DataFrame(shap_data["top_negative"])
                neg_df["shap_value"] = neg_df["shap_value"].apply(lambda x: f"{x:.4f}")
                st.dataframe(neg_df, use_container_width=True, hide_index=True)
            else:
                st.info("No negative SHAP features")

    else:
        st.info("💡 SHAP explanation unavailable. Run training notebooks to enable explanations.")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:rgba(255,255,255,0.4); font-size:0.8rem; padding: 20px;">
      🔍 Hybrid Fake News Detector · Powered by SBERT + Stylometric Features + SHAP · For Research Purposes Only
    </div>
    """, unsafe_allow_html=True)

elif not analyze_clicked:
    # Landing state — show feature highlights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 How This Works")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="section-card">
          <div class="section-title">🧠 Semantic Analysis</div>
          <p style="color:rgba(255,255,255,0.7); font-size:0.9rem;">
          Uses Sentence-BERT (all-MiniLM-L6-v2) to convert news articles into
          384-dimensional semantic vectors that capture meaning, context, and narrative style.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
          <div class="section-title">📐 Stylometric Analysis</div>
          <p style="color:rgba(255,255,255,0.7); font-size:0.9rem;">
          Analyzes writing patterns: sentiment polarity, subjectivity, readability scores,
          exclamation usage, capitalization patterns, and sentence structure.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="section-card">
          <div class="section-title">🔬 AI Explanation (SHAP)</div>
          <p style="color:rgba(255,255,255,0.7); font-size:0.9rem;">
          Provides transparent explanations showing which features influenced the
          prediction and by how much, making the AI decision interpretable.
          </p>
        </div>
        """, unsafe_allow_html=True)
