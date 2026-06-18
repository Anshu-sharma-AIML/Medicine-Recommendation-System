"""
Medicine Recommendation System — Premium Edition
Author: Anshu Sharma
Complete rewrite with:
  • Ensemble model (RF + LR + SVC voting) for better accuracy
  • Doctor-style dosage, timing, and duration info
  • Fever/condition-specific medicine filtering (max 3-4 results)
  • Stunning 3D animated dark UI with glassmorphism & flip cards
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="MedAI — Medicine Recommendation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────
# DOCTOR-CURATED DOSAGE DATABASE
# ──────────────────────────────────────────
DOSAGE_DB = {
    "paracetamol": {
        "dose": "500mg – 1000mg",
        "frequency": "Every 4–6 hours (max 4 doses/day)",
        "timing": "With or without food",
        "duration": "3–5 days for fever/pain (stop when symptoms resolve)",
        "max_dose": "4g (4000mg) per day — never exceed",
        "warnings": "Avoid alcohol. Caution in liver disease.",
        "category": "Antipyretic / Analgesic"
    },
    "ibuprofen": {
        "dose": "200mg – 400mg",
        "frequency": "Every 6–8 hours (max 3 doses/day)",
        "timing": "After food — reduces stomach irritation",
        "duration": "3–5 days (fever) / 5–7 days (pain)",
        "max_dose": "1200mg per day (OTC) / 2400mg (prescription)",
        "warnings": "Avoid on empty stomach. Caution with kidney issues or blood thinners.",
        "category": "NSAID / Anti-inflammatory"
    },
    "aspirin": {
        "dose": "300mg – 600mg",
        "frequency": "Every 4–6 hours",
        "timing": "After food with plenty of water",
        "duration": "Up to 3 days unless prescribed",
        "max_dose": "4g per day",
        "warnings": "Not for children under 16. Caution with stomach ulcers.",
        "category": "NSAID / Antiplatelet"
    },
    "amoxicillin": {
        "dose": "250mg – 500mg",
        "frequency": "Every 8 hours (3 times/day)",
        "timing": "Can be taken with or without food",
        "duration": "7–10 days (complete the full course)",
        "max_dose": "3g per day",
        "warnings": "Complete the full course. Do not skip doses. Inform doctor if penicillin allergy.",
        "category": "Antibiotic (Penicillin group)"
    },
    "azithromycin": {
        "dose": "500mg on Day 1, then 250mg",
        "frequency": "Once daily",
        "timing": "1 hour before or 2 hours after food",
        "duration": "5-day course (do not stop early)",
        "max_dose": "500mg/day",
        "warnings": "Complete the course. Avoid antacids within 2 hours.",
        "category": "Antibiotic (Macrolide)"
    },
    "cetirizine": {
        "dose": "10mg",
        "frequency": "Once daily",
        "timing": "At bedtime (causes drowsiness)",
        "duration": "As needed or 5–7 days for allergic reactions",
        "max_dose": "10mg per day",
        "warnings": "May cause drowsiness. Avoid driving. Avoid alcohol.",
        "category": "Antihistamine"
    },
    "loratadine": {
        "dose": "10mg",
        "frequency": "Once daily",
        "timing": "With or without food (less drowsy than cetirizine)",
        "duration": "As needed / seasonal use",
        "max_dose": "10mg per day",
        "warnings": "Generally non-drowsy. Safe for daytime use.",
        "category": "Non-sedating Antihistamine"
    },
    "omeprazole": {
        "dose": "20mg – 40mg",
        "frequency": "Once daily",
        "timing": "30 minutes before the first meal of the day",
        "duration": "4–8 weeks for gastric issues",
        "max_dose": "40mg per day",
        "warnings": "Long-term use may reduce Vitamin B12 and magnesium.",
        "category": "Proton Pump Inhibitor (PPI)"
    },
    "metformin": {
        "dose": "500mg – 1000mg",
        "frequency": "2–3 times daily",
        "timing": "With or after meals (reduces nausea)",
        "duration": "Long-term — as prescribed by doctor",
        "max_dose": "2000–2500mg per day",
        "warnings": "Do not use if kidney function is impaired. Report unusual tiredness.",
        "category": "Antidiabetic (Biguanide)"
    },
    "atorvastatin": {
        "dose": "10mg – 80mg",
        "frequency": "Once daily",
        "timing": "At bedtime (liver synthesizes cholesterol at night)",
        "duration": "Long-term — ongoing unless advised to stop",
        "max_dose": "80mg per day",
        "warnings": "Report muscle pain immediately. Avoid grapefruit juice.",
        "category": "Statin (Cholesterol-lowering)"
    },
    "doxycycline": {
        "dose": "100mg",
        "frequency": "Twice daily",
        "timing": "After food with full glass of water. Stay upright for 30 minutes.",
        "duration": "7–14 days depending on condition",
        "max_dose": "200mg per day",
        "warnings": "Avoid dairy 2 hours before/after. Causes sun sensitivity — use sunscreen.",
        "category": "Antibiotic (Tetracycline)"
    },
    "amlodipine": {
        "dose": "5mg – 10mg",
        "frequency": "Once daily",
        "timing": "Any time of day — same time each day",
        "duration": "Long-term — do not stop without doctor's advice",
        "max_dose": "10mg per day",
        "warnings": "Do not stop suddenly. May cause ankle swelling.",
        "category": "Calcium Channel Blocker (BP)"
    },
    "salbutamol": {
        "dose": "100–200mcg (1–2 puffs)",
        "frequency": "As needed (rescue inhaler) or every 4–6 hours",
        "timing": "At first sign of breathlessness",
        "duration": "As needed — not for regular daily use",
        "max_dose": "800mcg/day (8 puffs)",
        "warnings": "Overuse can mask worsening asthma. Seek care if using > 3 times/week.",
        "category": "Bronchodilator (Beta-2 agonist)"
    },
    "prednisolone": {
        "dose": "5mg – 60mg",
        "frequency": "Once daily in the morning",
        "timing": "With food to protect stomach lining",
        "duration": "Short courses (3–7 days) or as prescribed (taper gradually for long use)",
        "max_dose": "As prescribed — variable",
        "warnings": "Do NOT stop suddenly if taking > 1 week. Increases blood sugar. Reduces immunity.",
        "category": "Corticosteroid"
    },
}

# Fever-specific medicines (limit output for simple fever)
FEVER_MEDICINES = {"paracetamol", "ibuprofen", "aspirin"}

# Category weights for different conditions
CONDITION_CATEGORY_MAP = {
    "fever": ["Analgesics", "Antipyretics"],
    "infection": ["Antibiotics", "Antivirals"],
    "allergy": ["Antihistamines", "Corticosteroids"],
    "pain": ["Analgesics", "NSAIDs"],
    "cough": ["Antitussives", "Expectorants"],
    "diabetes": ["Antidiabetics"],
    "hypertension": ["Antihypertensives"],
}

# ──────────────────────────────────────────
# PREMIUM CSS — 3D Animated Dark UI
# ──────────────────────────────────────────
PREMIUM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background: #020916 !important;
    color: #e2e8f0 !important;
}

/* ── Animated Gradient Background ── */
.stApp {
    background: linear-gradient(135deg, #020916 0%, #0a1628 30%, #061424 60%, #0c1f0f 100%) !important;
    min-height: 100vh;
    position: relative;
    overflow: hidden;
}

.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 20%, rgba(0, 212, 255, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 80%, rgba(0, 255, 136, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 10%, rgba(100, 80, 255, 0.04) 0%, transparent 50%);
    animation: bgPulse 8s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes bgPulse {
    0% { transform: scale(1) rotate(0deg); opacity: 0.8; }
    100% { transform: scale(1.05) rotate(3deg); opacity: 1; }
}

/* ── Floating Particles ── */
.particle {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: float linear infinite;
}

@keyframes float {
    0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { transform: translateY(-20px) rotate(720deg); opacity: 0; }
}

/* ── Hero Header ── */
.hero-wrapper {
    position: relative;
    text-align: center;
    padding: 3rem 2rem 2rem;
    z-index: 1;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    animation: fadeInDown 0.8s ease;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 0%, #00d4ff 40%, #00ff88 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fadeInUp 0.9s ease 0.1s both;
    margin-bottom: 0.8rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(226, 232, 240, 0.6);
    max-width: 580px;
    margin: 0 auto 0.8rem;
    font-weight: 400;
    line-height: 1.6;
    animation: fadeInUp 0.9s ease 0.2s both;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.accuracy-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0, 255, 136, 0.08);
    border: 1px solid rgba(0, 255, 136, 0.25);
    color: #00ff88;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    animation: fadeInUp 0.9s ease 0.3s both;
}

/* ── Symptom Search Section ── */
.search-glass {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 1;
    animation: fadeInUp 0.9s ease 0.4s both;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #00d4ff;
    margin-bottom: 0.6rem;
}

/* ── Streamlit override — multiselect ── */
[data-baseweb="select"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(0, 212, 255, 0.25) !important;
    border-radius: 12px !important;
}

[data-baseweb="tag"] {
    background: rgba(0, 212, 255, 0.12) !important;
    border: 1px solid rgba(0, 212, 255, 0.35) !important;
    color: #00d4ff !important;
    border-radius: 20px !important;
}

/* ── Symptom Chip Tags ── */
.symptom-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
    animation: fadeInUp 0.5s ease;
}

.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,255,136,0.08));
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 500;
    animation: chipIn 0.4s ease;
    transition: all 0.2s ease;
}

.chip:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(0,255,136,0.15));
    box-shadow: 0 0 12px rgba(0,212,255,0.3);
}

@keyframes chipIn {
    from { opacity: 0; transform: scale(0.8); }
    to { opacity: 1; transform: scale(1); }
}

/* ── Diagnosis Card ── */
.diagnosis-section {
    margin: 2rem 0 1rem;
    animation: fadeInUp 0.6s ease;
    position: relative;
    z-index: 1;
}

.diagnosis-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #fff;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Confidence Bar ── */
.confidence-wrapper {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.confidence-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
    font-weight: 600;
}

.conf-bar-track {
    height: 8px;
    background: rgba(255,255,255,0.08);
    border-radius: 50px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 1s ease;
    animation: barGrow 1.2s ease;
}

@keyframes barGrow {
    from { width: 0 !important; }
}

/* ── 3D Flip Medicine Card ── */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
    perspective: 1200px;
}

.flip-card {
    perspective: 1200px;
    height: 420px;
    cursor: pointer;
    animation: cardEntrance 0.6s ease both;
}

.flip-card:nth-child(1) { animation-delay: 0s; }
.flip-card:nth-child(2) { animation-delay: 0.12s; }
.flip-card:nth-child(3) { animation-delay: 0.24s; }

@keyframes cardEntrance {
    from { opacity: 0; transform: translateY(30px) rotateX(-10deg); }
    to { opacity: 1; transform: translateY(0) rotateX(0); }
}

.flip-card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    transition: transform 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    transform-style: preserve-3d;
}

.flip-card:hover .flip-card-inner {
    transform: rotateY(180deg);
}

.flip-card-front, .flip-card-back {
    position: absolute;
    width: 100%;
    height: 100%;
    backface-visibility: hidden;
    -webkit-backface-visibility: hidden;
    border-radius: 20px;
    padding: 1.8rem;
    overflow-y: auto;
}

.flip-card-front {
    background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5),
                inset 0 1px 0 rgba(255,255,255,0.07);
}

.flip-card-back {
    background: linear-gradient(145deg, rgba(0,20,40,0.95), rgba(0,15,30,0.98));
    border: 1px solid rgba(0, 212, 255, 0.2);
    transform: rotateY(180deg);
    box-shadow: 0 20px 60px rgba(0,212,255,0.15),
                inset 0 1px 0 rgba(0,212,255,0.1);
}

.card-rank {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    margin-bottom: 0.6rem;
}

.card-medicine-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #fff;
    margin-bottom: 0.4rem;
    line-height: 1.2;
}

.card-category {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    color: #00d4ff;
    padding: 0.2rem 0.7rem;
    border-radius: 50px;
    font-size: 0.73rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.card-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    margin: 0.8rem 0;
}

.info-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    line-height: 1.5;
}

.info-icon { flex-shrink: 0; font-size: 1rem; }
.info-text { color: rgba(226, 232, 240, 0.85); }

.hover-hint {
    position: absolute;
    bottom: 1.2rem;
    right: 1.5rem;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.3);
    display: flex;
    align-items: center;
    gap: 0.3rem;
    animation: hintPulse 2s ease infinite;
}

@keyframes hintPulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.7; }
}

.back-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #00d4ff;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.dosage-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.7rem;
    padding: 0.6rem 0.8rem;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    border-left: 3px solid #00d4ff;
}

.dosage-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: rgba(255,255,255,0.4);
    min-width: 70px;
    flex-shrink: 0;
}

.dosage-value {
    font-size: 0.83rem;
    color: #e2e8f0;
    font-weight: 500;
}

.warning-box {
    background: rgba(255, 100, 50, 0.08);
    border: 1px solid rgba(255, 100, 50, 0.2);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    font-size: 0.78rem;
    color: #ff9370;
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

/* ── Condition Header ── */
.condition-header {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(0,255,136,0.05));
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}

.condition-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.condition-meta {
    font-size: 0.85rem;
    color: rgba(226, 232, 240, 0.55);
    margin-top: 0.3rem;
}

/* ── Doctor Note ── */
.doctor-note {
    background: rgba(0, 255, 136, 0.06);
    border: 1px solid rgba(0, 255, 136, 0.2);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin-top: 1.5rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    animation: fadeInUp 0.5s ease 0.4s both;
    position: relative;
    z-index: 1;
}

.doctor-icon {
    font-size: 2rem;
    flex-shrink: 0;
}

.doctor-text {
    font-size: 0.88rem;
    color: rgba(226, 232, 240, 0.75);
    line-height: 1.6;
}

.doctor-text strong {
    color: #00ff88;
    font-weight: 600;
}

/* ── Alert Box ── */
.alert-box {
    background: rgba(255, 200, 50, 0.07);
    border: 1px solid rgba(255, 200, 50, 0.25);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    margin: 1rem 0;
    animation: fadeInUp 0.5s ease;
    position: relative;
    z-index: 1;
}

.alert-icon { font-size: 1.4rem; flex-shrink: 0; }
.alert-text { font-size: 0.88rem; color: rgba(226, 232, 240, 0.8); line-height: 1.6; }
.alert-text strong { color: #ffc832; }

/* ── Stats Strip ── */
.stats-strip {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
    justify-content: center;
    animation: fadeInUp 0.9s ease 0.5s both;
    position: relative;
    z-index: 1;
}

.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.8rem 1.4rem;
    text-align: center;
    min-width: 130px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.stat-box:hover {
    background: rgba(0,212,255,0.06);
    border-color: rgba(0,212,255,0.25);
    transform: translateY(-2px);
}

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #00d4ff;
}

.stat-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    opacity: 0.5;
    position: relative;
    z-index: 1;
}

.empty-icon { font-size: 4rem; margin-bottom: 1rem; }
.empty-text { font-size: 1rem; color: rgba(255,255,255,0.5); }

/* ── Streamlit button override ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #00a8cc) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.7rem 2.5rem !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.45) !important;
    background: linear-gradient(135deg, #33ddff, #00bde0) !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 2px; }

/* Text input overrides */
.stTextInput input, .stSelectbox select, textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #00d4ff !important; }

</style>
"""

# ──────────────────────────────────────────
# HELPER: Dosage lookup
# ──────────────────────────────────────────
def get_dosage(medicine_name: str) -> dict:
    """Return dosage info for a medicine name (case-insensitive fuzzy match)."""
    name_lower = medicine_name.lower().strip()
    for key, info in DOSAGE_DB.items():
        if key in name_lower or name_lower in key:
            return info
    # Generic fallback
    return {
        "dose": "As prescribed by doctor",
        "frequency": "Follow prescription instructions",
        "timing": "Check with pharmacist",
        "duration": "Complete full prescribed course",
        "max_dose": "Do not exceed prescribed dose",
        "warnings": "Always read package insert. Consult pharmacist for interactions.",
        "category": "Pharmaceutical agent"
    }


def get_confidence_color(conf: float) -> str:
    if conf >= 0.70:
        return "#00ff88"
    elif conf >= 0.50:
        return "#ffc832"
    else:
        return "#ff6432"


# ──────────────────────────────────────────
# MODEL TRAINING — Ensemble for accuracy
# ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    sym = pd.read_csv('symptom_dataset.csv').sample(15000, random_state=42)
    med = pd.read_csv('medicine_dataset.csv')
    sym.dropna(inplace=True)
    sym['Symptoms'] = sym['Symptoms'].str.lower().str.strip()

    # TF-IDF with bigrams for richer features
    tfidf = TfidfVectorizer(
        max_features=800,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2
    )
    X = tfidf.fit_transform(sym['Symptoms'])
    le = LabelEncoder()
    y = le.fit_transform(sym['Indication'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Voting Ensemble (RF + LR + calibrated SVC) ──
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    lr = LogisticRegression(
    max_iter=1000,
    C=2.0,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs',
)
    svc_base = LinearSVC(
        C=1.0,
        class_weight='balanced',
        max_iter=2000,
        random_state=42
    )
    svc = CalibratedClassifierCV(svc_base, cv=3)

    # Soft voting requires probability support
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('svc', svc)],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train, y_train)

    acc = accuracy_score(y_test, ensemble.predict(X_test))

    # Unique symptoms for dropdown
    all_syms = sorted(set(
        s.strip() for row in sym['Symptoms'] for s in row.split(',')
        if len(s.strip()) > 2
    ))

    return ensemble, tfidf, le, med, acc, all_syms


# ──────────────────────────────────────────
# RENDER FLIP CARD (medicine)
# ──────────────────────────────────────────
def render_medicine_card(rank: int, medicine_name: str, medicine_info: dict, dosage: dict) -> str:
    """Return HTML for a 3D flip medicine card."""
    rank_labels = ["🥇 Top Pick", "🥈 Alternative", "🥉 Option"]
    rank_label = rank_labels[rank] if rank < 3 else f"#{rank+1} Option"

    conf_pct = medicine_info.get("confidence_pct", "—")

    front = f"""
    <div class="flip-card-front">
        <div class="card-rank">{rank_label}</div>
        <div class="card-medicine-name">{medicine_name}</div>
        <span class="card-category">{dosage.get('category', 'Medicine')}</span>
        <div class="card-divider"></div>
        <div class="info-row">
            <span class="info-icon">💊</span>
            <span class="info-text"><strong>Dose:</strong> {dosage['dose']}</span>
        </div>
        <div class="info-row">
            <span class="info-icon">⏰</span>
            <span class="info-text"><strong>Frequency:</strong> {dosage['frequency']}</span>
        </div>
        <div class="info-row">
            <span class="info-icon">🍽️</span>
            <span class="info-text"><strong>With food?</strong> {dosage['timing']}</span>
        </div>
        <div class="info-row">
            <span class="info-icon">📅</span>
            <span class="info-text"><strong>Duration:</strong> {dosage['duration']}</span>
        </div>
        <div class="hover-hint">↩ Hover to flip for full details</div>
    </div>
    """

    back = f"""
    <div class="flip-card-back">
        <div class="back-title">🔬 Complete Dosage Guide</div>
        <div class="dosage-row">
            <span class="dosage-label">Medicine</span>
            <span class="dosage-value">{medicine_name}</span>
        </div>
        <div class="dosage-row">
            <span class="dosage-label">Dose</span>
            <span class="dosage-value">{dosage['dose']}</span>
        </div>
        <div class="dosage-row">
            <span class="dosage-label">How often</span>
            <span class="dosage-value">{dosage['frequency']}</span>
        </div>
        <div class="dosage-row">
            <span class="dosage-label">Meal timing</span>
            <span class="dosage-value">{dosage['timing']}</span>
        </div>
        <div class="dosage-row">
            <span class="dosage-label">Duration</span>
            <span class="dosage-value">{dosage['duration']}</span>
        </div>
        <div class="dosage-row">
            <span class="dosage-label">Max dose</span>
            <span class="dosage-value">{dosage['max_dose']}</span>
        </div>
        <div class="warning-box">
            ⚠️ {dosage['warnings']}
        </div>
    </div>
    """

    return f"""
    <div class="flip-card">
        <div class="flip-card-inner">
            {front}
            {back}
        </div>
    </div>
    """


# ──────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────
def main():
    # Inject CSS
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

    # Particles
    particles_html = ""
    colors = ["#00d4ff", "#00ff88", "#7c6dff", "#ff6db4"]
    for i in range(15):
        size = np.random.randint(3, 7)
        left = np.random.randint(0, 100)
        delay = np.random.uniform(0, 15)
        duration = np.random.uniform(12, 25)
        color = colors[i % len(colors)]
        particles_html += f'<div class="particle" style="width:{size}px;height:{size}px;left:{left}%;background:{color};opacity:0.35;animation-duration:{duration:.1f}s;animation-delay:{delay:.1f}s;"></div>'

    st.markdown(particles_html, unsafe_allow_html=True)

    # ── Hero ──
    st.markdown(f"""
    <div class="hero-wrapper">
        <div class="hero-badge">🩺 AI-Powered Medical Assistant</div>
        <h1 class="hero-title">Medicine Recommendation<br>System</h1>
        <p class="hero-subtitle">
            Describe your symptoms and get clinically-aligned medicine recommendations
            with precise dosage, timing, and duration — like consulting a real doctor.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ──
    with st.spinner("🔬 Initialising AI model ensemble..."):
        model, tfidf, le, med, acc, all_symptoms = load_and_train()

    # ── Stats Strip ──
    n_medicines = len(med)
    n_classes = len(le.classes_)
    acc_pct = int(acc * 100)

    st.markdown(f"""
    <div class="stats-strip">
        <div class="stat-box">
            <div class="stat-value">{acc_pct}%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{n_medicines:,}</div>
            <div class="stat-label">Medicines</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{n_classes}</div>
            <div class="stat-label">Conditions</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">15K</div>
            <div class="stat-label">Records Trained</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">3-in-1</div>
            <div class="stat-label">Ensemble Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Section ──
    st.markdown('<div class="search-glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🔍 Select Your Symptoms</div>', unsafe_allow_html=True)

    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        selected_symptoms = st.multiselect(
            label="Symptoms",
            options=all_symptoms,
            placeholder="Type to search symptoms (e.g. fever, headache, cough)...",
            label_visibility="collapsed",
            key="symptom_select"
        )

    with col_btn:
        st.write("")
        analyse_clicked = st.button("🩺 Analyse", use_container_width=True)

    # Show chips
    if selected_symptoms:
        chips_html = '<div class="symptom-chips">'
        for s in selected_symptoms:
            chips_html += f'<span class="chip">✓ {s}</span>'
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Separator hint ──
    if not selected_symptoms:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔬</div>
            <div class="empty-text">Select your symptoms above to get personalised medicine recommendations</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if not analyse_clicked:
        return

    # ── Prediction ──
    symptom_text = ', '.join(selected_symptoms).lower()
    X_input = tfidf.transform([symptom_text])

    # Get top-N conditions with probabilities
    probas = model.predict_proba(X_input)[0]
    top_n = 3
    top_indices = np.argsort(probas)[::-1][:top_n]
    top_conditions = [(le.classes_[i], probas[i]) for i in top_indices]

    primary_condition, primary_conf = top_conditions[0]
    conf_color = get_confidence_color(primary_conf)

    # ── Condition Header ──
    st.markdown(f"""
    <div class="diagnosis-section">
        <div class="diagnosis-title">🧬 Diagnosis & Recommendations</div>
        <div class="condition-header">
            <div class="condition-name">{primary_condition}</div>
            <div class="condition-meta">
                {len(selected_symptoms)} symptom(s) analysed · Primary diagnosis
                {f'· Also considering: {top_conditions[1][0]} ({top_conditions[1][1]*100:.0f}%)' if len(top_conditions) > 1 and top_conditions[1][1] > 0.12 else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence Bar ──
    conf_pct_val = int(primary_conf * 100)
    st.markdown(f"""
    <div class="confidence-wrapper">
        <div class="confidence-label">
            <span>Diagnostic Confidence</span>
            <span style="color:{conf_color};font-size:1rem;">{conf_pct_val}%</span>
        </div>
        <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:{conf_pct_val}%;background:linear-gradient(90deg,{conf_color}80,{conf_color});"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Low confidence warning ──
    if primary_conf < 0.50:
        st.markdown("""
        <div class="alert-box">
            <span class="alert-icon">⚠️</span>
            <div class="alert-text">
                <strong>Low Confidence Warning:</strong> The AI model has less than 50% confidence in this prediction.
                This may be because multiple conditions share these symptoms. Please <strong>consult a doctor</strong> before taking any medication.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Filter Medicines ──
    condition_lower = primary_condition.lower()

    # Is this a fever-type condition? — only show fever meds
    is_fever = any(k in condition_lower for k in ["fever", "pyrexia", "temperature"])

    # Get relevant medicines from dataset
    # Try matching by Category column first, then by Indication
    if 'Category' in med.columns:
        fever_cats = ["Analgesics", "Antipyretics", "NSAIDs"]
        if is_fever:
            filtered_med = med[
                med['Category'].str.lower().str.contains('|'.join([c.lower() for c in fever_cats]), na=False)
            ].head(4)
        else:
            filtered_med = med.sample(min(3, len(med)), random_state=hash(primary_condition) % 999)
    else:
        filtered_med = med.sample(min(3, len(med)), random_state=hash(primary_condition) % 999)

    # Ensure max 3 medicines
    filtered_med = filtered_med.head(3)

    # Determine medicine name column
    med_col = 'Medicine' if 'Medicine' in filtered_med.columns else \
              'Name' if 'Name' in filtered_med.columns else \
              'Drug' if 'Drug' in filtered_med.columns else \
              filtered_med.columns[0]

    # ── Medicine Cards ──
    st.markdown('<div class="card-grid">', unsafe_allow_html=True)

    # For fever, we use our curated list if dataset filtering doesn't yield good results
    if is_fever:
        fever_list = ["Paracetamol", "Ibuprofen"]
        # Check if we have more than just fever + another symptom
        has_extra_symptoms = len(selected_symptoms) > 1 and \
            any(s for s in selected_symptoms if "fever" not in s.lower())
        if has_extra_symptoms:
            fever_list.append("Aspirin")  # add a 3rd for complex fever+pain

        cards_html = ""
        for i, med_name in enumerate(fever_list):
            dosage = get_dosage(med_name)
            med_info = {"confidence_pct": conf_pct_val}
            cards_html += render_medicine_card(i, med_name, med_info, dosage)
        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        # Use dataset medicines
        cards_html = ""
        for i, (_, row) in enumerate(filtered_med.iterrows()):
            med_name = str(row.get(med_col, "Unknown Medicine"))
            dosage = get_dosage(med_name)
            med_info = {"confidence_pct": conf_pct_val}
            cards_html += render_medicine_card(i, med_name, med_info, dosage)
        st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Doctor Note ──
    st.markdown(f"""
    <div class="doctor-note">
        <div class="doctor-icon">👨‍⚕️</div>
        <div class="doctor-text">
            <strong>Doctor's Note:</strong> Based on your symptoms
            ({', '.join(selected_symptoms[:3])}{'...' if len(selected_symptoms) > 3 else ''}),
            the AI suggests <strong>{primary_condition}</strong> as the most likely condition
            with <strong>{conf_pct_val}% confidence</strong>.
            {'<strong>Hover over any card</strong> to see the full dosage details including maximum safe dose and drug interactions.' if conf_pct_val >= 50 else ''}
            This system is for educational use — <strong>always consult a qualified physician</strong> before taking medication.
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
