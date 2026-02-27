import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Medicine Recommendation System",
    page_icon="💊",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #2E86C1;
}
.sub-title {
    font-size: 18px;
    color: #555;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #F8F9F9;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💊 Medicine Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-based system to recommend medicine based on symptoms</div>', unsafe_allow_html=True)
st.write("---")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("./dataset.csv")

X = df["Indication"]
y = df["Name"]

vectorizer = CountVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# ---------------- MEDICINE DOSE DATABASE ----------------
medicine_info = {
    "Paracetamol": ("500 mg", "2 times a day", "3 days"),
    "Ibuprofen": ("400 mg", "2 times a day", "3 days"),
    "Diclofenac": ("50 mg", "2 times a day", "3 days"),
    "ORS": ("1 sachet", "after each loose motion", "1–2 days"),
    "Ondansetron": ("4 mg", "1–2 times a day", "2 days"),
    "Cetirizine": ("10 mg", "once at night", "3 days"),
    "Azithromycin": ("500 mg", "once a day", "3 days"),
    "Amoxicillin": ("500 mg", "2 times a day", "5 days"),
    "Pantoprazole": ("40 mg", "once before breakfast", "5 days"),
    "Lactulose": ("15 ml", "once at night", "3 days"),
    "Salbutamol": ("2 puffs", "when needed", "as advised"),
    "Amlodipine": ("5 mg", "once daily", "as advised"),
    "Metformin": ("500 mg", "2 times a day", "long term"),
    "Dicycloverine": ("20 mg", "2 times a day", "3 days"),
}

# ---------------- INPUT SECTION ----------------
st.markdown("### 📝 Enter Symptoms")

symptoms = st.text_input(
    "",
    placeholder="Example: fever, headache, loose motion"
)

# ---------------- PREDICTION ----------------
if st.button("🔍 Recommend Medicine"):
    if symptoms.strip() == "":
        st.warning("Please enter symptoms")
    else:
        input_vec = vectorizer.transform([symptoms])
        medicine = model.predict(input_vec)[0]

        dose, frequency, duration = medicine_info.get(
            medicine,
            ("Consult doctor", "Consult doctor", "Consult doctor")
        )

        # ---------------- RESULT CARD ----------------
        st.write("")
        st.markdown(f"""
        <div class="card">
            <h3>✅ Recommended Medicine</h3>
            <p><b>Medicine Name:</b> {medicine}</p>
            <p><b>Dosage:</b> {dose}</p>
            <p><b>Frequency:</b> {frequency}</p>
            <p><b>Duration:</b> {duration}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.write("---")
st.caption("⚠️ This system is for educational purposes only. Please consult a doctor before taking any medication.")

