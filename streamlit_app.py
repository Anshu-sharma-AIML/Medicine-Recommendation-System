import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Medicine Recommendation System",
    page_icon="💊",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.title { font-family: 'DM Serif Display', serif; font-size: 2.5rem; text-align: center; }
.symptom-tag {
    display: inline-block;
    background: #1a3a2a;
    border: 1px solid #2ecc71;
    color: #2ecc71;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    margin: 0.2rem;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train():
    sym = pd.read_csv('symptom_dataset.csv').sample(15000, random_state=42)
    med = pd.read_csv('medicine_dataset.csv')
    sym.dropna(inplace=True)
    sym['Symptoms'] = sym['Symptoms'].str.lower().str.strip()

    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X     = tfidf.fit_transform(sym['Symptoms'])
    le    = LabelEncoder()
    y     = le.fit_transform(sym['Indication'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, model.predict(X_test))

    # Get all unique symptoms
    all_syms = sorted(set(
        s.strip() for row in sym['Symptoms'] for s in row.split(',')
    ))

    return model, tfidf, le, med, acc, all_syms


with st.spinner("Loading model..."):
    model, tfidf, le, med, acc, all_symptoms = load_and_train()

# ── Header ──────────────────────────────────────────
st.markdown('<div class="title">💊 Medicine Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#888;font-size:0.9rem;">AI-based system to recommend medicines based on symptoms · 50,000 records</p>', unsafe_allow_html=True)

st.divider()

# Stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset", "50,000", "records")
col2.metric("Model Accuracy", f"{acc:.1%}", "Random Forest")
col3.metric("Disease Classes", "8", "indications")
col4.metric("Unique Symptoms", str(len(all_symptoms)), "features")

st.divider()

# ── Input ────────────────────────────────────────────
st.subheader("🩺 Select Your Symptoms")

selected = st.multiselect(
    "Choose all symptoms you are experiencing:",
    options=all_symptoms,
    default=["fever", "headache"]
)

st.caption("Or type custom symptoms below (comma separated):")
custom = st.text_input("Custom symptoms", placeholder="e.g. fever, headache, fatigue")

st.divider()

if st.button("🔍 Get Medicine Recommendations", use_container_width=True, type="primary"):
    # Combine selected + custom
    all_input = list(selected)
    if custom.strip():
        all_input.extend([s.strip() for s in custom.split(',')])

    if not all_input:
        st.warning("Please select or enter at least one symptom!")
    else:
        symptoms_str = ', '.join(all_input)
        X_input    = tfidf.transform([symptoms_str.lower()])
        indication = le.inverse_transform(model.predict(X_input))[0]

        # Get probabilities for confidence
        if hasattr(model, 'predict_proba'):
            proba    = model.predict_proba(X_input)[0]
            top3_idx = np.argsort(proba)[::-1][:3]
            confidence = proba[top3_idx[0]] * 100
        else:
            confidence = None

        st.success(f"### 🎯 Predicted Condition: **{indication}**" +
                   (f" (Confidence: {confidence:.1f}%)" if confidence else ""))

        # Show selected symptoms
        st.markdown("**Your symptoms:** " +
                    " ".join([f'<span class="symptom-tag">{s}</span>' for s in all_input]),
                    unsafe_allow_html=True)

        st.divider()

        # Recommend medicines
        st.subheader(f"💊 Recommended Medicines for {indication}")

        candidates = med[med['Indication'] == indication]
        if len(candidates) > 0:
            recs = candidates.sample(min(6, len(candidates)), random_state=42)

            for _, row in recs.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 2])
                    with col1:
                        st.markdown(f"**{row['Name']}**")
                        st.caption(f"{row['Category']}")
                    with col2:
                        st.markdown(f"📋 {row['Dosage Form']}")
                        st.caption(f"Strength: {row['Strength']}")
                    with col3:
                        color = "🟢" if row['Classification'] == 'Over-the-Counter' else "🔴"
                        st.markdown(f"{color} {row['Classification']}")
                    st.divider()

        # Show top 3 possible conditions
        if confidence and hasattr(model, 'predict_proba'):
            st.subheader("📊 Condition Probabilities")
            for idx in top3_idx:
                cond = le.classes_[idx]
                prob = proba[idx] * 100
                st.progress(prob / 100, text=f"{cond}: {prob:.1f}%")

    st.divider()

st.warning("⚠️ This system is for educational purposes only. Always consult a qualified doctor before taking any medication.")
st.caption("Built by Anshu Sharma · B.Tech CSE · Shri Shankaracharya Professional University, Bhilai")
