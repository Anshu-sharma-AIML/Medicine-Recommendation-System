"""
Medicine Recommendation System
Author : Anshu Sharma
Dataset: 50,000 real medicine & symptom records
Tech   : Python, Pandas, NumPy, Scikit-learn, TF-IDF, NLP, Matplotlib, Seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')


# ─────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────
def load_data(sym_path='symptom_dataset.csv', med_path='medicine_dataset.csv',
              sample=15000, seed=42):
    print("[1/5] Loading datasets...")
    sym = pd.read_csv(sym_path).sample(sample, random_state=seed)
    med = pd.read_csv(med_path)
    sym.dropna(inplace=True)
    sym['Symptoms'] = sym['Symptoms'].str.lower().str.strip()
    print(f"      Symptom data : {sym.shape[0]:,} rows")
    print(f"      Medicine data: {med.shape[0]:,} rows")
    print(f"      Indications  : {sym['Indication'].nunique()} classes")
    print(f"      Unique symptoms: {len(set(s for row in sym['Symptoms'] for s in row.split(',')))}")
    return sym, med


# ─────────────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────────────
def run_eda(sym, med):
    print("[2/5] Running EDA...")

    # Indication distribution
    plt.figure(figsize=(10, 6))
    sym['Indication'].value_counts().plot(kind='bar', color='steelblue', edgecolor='white')
    plt.title('Disease Indication Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Indication')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/indication_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Medicine categories
    plt.figure(figsize=(10, 6))
    med['Category'].value_counts().plot(kind='bar', color='coral', edgecolor='white')
    plt.title('Medicine Categories Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/medicine_categories.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Classification pie
    plt.figure(figsize=(8, 5))
    med['Classification'].value_counts().plot(
        kind='pie', autopct='%1.1f%%',
        colors=['#3498db', '#e74c3c'], startangle=90)
    plt.title('Prescription vs OTC Medicines', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/classification_pie.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Top symptoms
    all_syms = []
    for s in sym['Symptoms']:
        all_syms.extend([x.strip() for x in s.split(',')])
    sym_freq = pd.Series(all_syms).value_counts().head(15)

    plt.figure(figsize=(10, 6))
    sym_freq.plot(kind='barh', color='mediumpurple')
    plt.title('Top 15 Most Common Symptoms', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('plots/symptom_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("      Plots saved to /plots/")


# ─────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────
def build_features(sym):
    print("[3/5] Building features with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X     = tfidf.fit_transform(sym['Symptoms'])
    le    = LabelEncoder()
    y     = le.fit_transform(sym['Indication'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"      Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"      Features (TF-IDF): {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, tfidf, le


# ─────────────────────────────────────────────────────
# 4. TRAIN & EVALUATE
# ─────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test, le):
    print("[4/5] Training & evaluating models...")

    model_list = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Random Forest',       RandomForestClassifier(n_estimators=100, random_state=42)),
        ('Naive Bayes',         MultinomialNB()),
        ('Linear SVC',          LinearSVC(random_state=42, max_iter=2000)),
    ]

    results = []
    for name, model in model_list:
        model.fit(X_train, y_train)
        yp  = model.predict(X_test)
        acc = accuracy_score(y_test, yp)
        print(f"      {name:<25} Accuracy = {acc:.4f}")
        results.append({'name': name, 'model': model, 'acc': acc, 'preds': yp})

    # Comparison plot
    res_df = pd.DataFrame([{'Model': r['name'], 'Accuracy': r['acc']} for r in results])
    res_df = res_df.sort_values('Accuracy', ascending=True)
    colors = ['#2ecc71' if v == res_df['Accuracy'].max() else '#3498db'
              for v in res_df['Accuracy']]

    plt.figure(figsize=(10, 6))
    plt.barh(res_df['Model'], res_df['Accuracy'], color=colors)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1.1)
    for i, v in enumerate(res_df['Accuracy']):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Best model confusion matrix
    best = max(results, key=lambda x: x['acc'])
    cm   = confusion_matrix(y_test, best['preds'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix — {best["name"]}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n      Best Model: {best['name']} — Accuracy: {best['acc']:.4f}")
    print(f"\n      Classification Report ({best['name']}):")
    print(classification_report(y_test, best['preds'], target_names=le.classes_))

    return best['model'], results


# ─────────────────────────────────────────────────────
# 5. RECOMMEND MEDICINES
# ─────────────────────────────────────────────────────
def recommend(symptoms_input, model, tfidf, le, med_df, top_n=5):
    """Given a symptom string, predict indication and recommend medicines."""
    symptoms_clean = symptoms_input.lower().strip()
    X_input    = tfidf.transform([symptoms_clean])
    indication = le.inverse_transform(model.predict(X_input))[0]

    recommendations = (med_df[med_df['Indication'] == indication]
                       .sample(min(top_n, len(med_df[med_df['Indication'] == indication])),
                               random_state=42)
                       [['Name', 'Category', 'Dosage Form', 'Strength', 'Classification']])
    return indication, recommendations


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  MEDICINE RECOMMENDATION SYSTEM")
    print("="*55 + "\n")

    sym, med = load_data()
    run_eda(sym, med)
    X_train, X_test, y_train, y_test, tfidf, le = build_features(sym)
    best_model, results = train_models(X_train, X_test, y_train, y_test, le)

    print("\n[5/5] Sample Predictions...")
    test_cases = [
        "fever, headache, tiredness",
        "cough, sore throat, body ache",
        "frequent urination, thirst, fatigue",
        "redness, swelling, pus",
    ]

    for symptoms in test_cases:
        indication, recs = recommend(symptoms, best_model, tfidf, le, med)
        print(f"\n{'─'*50}")
        print(f"  Symptoms  : {symptoms}")
        print(f"  Diagnosis : {indication}")
        print(f"  Recommended Medicines:")
        for _, row in recs.iterrows():
            print(f"    • {row['Name']} ({row['Category']}) — {row['Dosage Form']} {row['Strength']}")

    print(f"\n{'='*55}")
    print("✅ Done! Check /plots/ for all visualisations.")
