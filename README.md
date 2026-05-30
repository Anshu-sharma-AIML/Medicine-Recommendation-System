# 💊 Medicine Recommendation System

An AI-powered system that recommends medicines based on patient symptoms using **NLP and Machine Learning classification models**.

**Tech Stack:** Python · Pandas · NumPy · Scikit-learn · TF-IDF · NLP · Matplotlib · Seaborn · Streamlit

---

## 🚀 Live Demo
👉 **[Click here to try the app](https://your-streamlit-link-here.streamlit.app)**

---

## 📌 Project Overview

This project builds a complete ML pipeline to:
1. **Process symptoms** using NLP (TF-IDF vectorization)
2. **Classify the disease** indication from symptoms (8 classes)
3. **Recommend medicines** with dosage, strength and classification details

---

## 📂 Project Structure

```
medicine-recommendation-system/
│
├── symptom_dataset.csv          # 50,000 symptom-indication pairs
├── medicine_dataset.csv         # 50,000 medicine records
├── model.py                     # Full ML pipeline script
├── streamlit_app.py             # Interactive web application
├── requirements.txt             # Dependencies
├── plots/                       # EDA and model plots
│   ├── indication_distribution.png
│   ├── medicine_categories.png
│   ├── classification_pie.png
│   ├── symptom_frequency.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
└── README.md
```

---

## 📊 Dataset

| Dataset | Rows | Description |
|---------|------|-------------|
| `symptom_dataset.csv` | 50,000 | Symptom-to-indication mappings |
| `medicine_dataset.csv` | 50,000 | Medicine details (name, category, dosage, strength) |

**Disease Classes:** Infection · Virus · Fever · Pain · Diabetes · Depression · Fungus · Wound

**Symptoms (19 unique):** fever, headache, cough, fatigue, body ache, sore throat, redness, swelling, frequent urination, thirst, muscle pain, bleeding, pus, malaise, tiredness, discomfort, stiffness, general weakness, pain

---

## 🔧 NLP Pipeline

```
Raw Symptoms Text
      ↓
Text Preprocessing (lowercase, strip)
      ↓
TF-IDF Vectorization (500 features, bigrams)
      ↓
Classification Model
      ↓
Predicted Indication
      ↓
Medicine Recommendations
```

---

## 🤖 Models Trained & Results

| Model | Accuracy |
|-------|----------|
| **Random Forest** | **~0.69** ⭐ |
| Linear SVC | ~0.69 |
| Logistic Regression | ~0.69 |
| Naive Bayes | ~0.68 |

---

## 📈 Key Features

- Multi-symptom input (select from dropdown or type custom)
- Disease confidence probability display
- Top 3 possible conditions shown
- Medicine details: name, category, dosage form, strength, OTC vs prescription
- 6 EDA visualisation plots

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run ML pipeline
python model.py

# Run web app
streamlit run streamlit_app.py
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas & NumPy** — data processing
- **Scikit-learn** — ML models, TF-IDF, evaluation
- **NLP** — TF-IDF vectorization, text preprocessing
- **Matplotlib & Seaborn** — visualisation
- **Streamlit** — web application

---

## 👤 Author

**Anshu Sharma**  
B.Tech CSE — Shri Shankaracharya Professional University, Bhilai  
📧 anshusharma6117@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/anshu-sharma13)
