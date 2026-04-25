import numpy as np
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Crop Advisor AI 🌾",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "predicted" not in st.session_state:
    st.session_state.predicted = False

if "last_lang" not in st.session_state:
    st.session_state.last_lang = "English"

# ---------------- ENV ----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-flash-lite-latest")
else:
    gemini_model = None

# ---------------- TOP BAR (LANGUAGE RIGHT) ----------------
col1, col2 = st.columns([8, 1])

with col2:
    lang = st.selectbox(
        "",
        ["English", "Hindi", "Marathi"],
        label_visibility="collapsed"
    )

# ---------------- LANGUAGE LOGIC ----------------
lang_map = {
    "English": "Respond in English",
    "Hindi": "Respond in Hindi",
    "Marathi": "Respond in Marathi"
}

lang_instruction = lang_map[lang]

# Detect language change after prediction
if st.session_state.predicted and lang != st.session_state.last_lang:
    st.warning("⚠️ Please click 'Predict Crop' again to apply language change.")

# ---------------- SAFE GEMINI ----------------
def safe_gemini(prompt, fallback):
    try:
        if gemini_model is None:
            return fallback
        res = gemini_model.generate_content(prompt + f"\n{lang_instruction}")
        return res.text if res and res.text else fallback
    except:
        return fallback

# ---------------- AI FUNCTIONS ----------------
def simple_reason(crop):
    return safe_gemini(
        f"Explain in simple bullet points why {crop} is suitable for farmer.",
        f"{crop} suits your soil and weather."
    )

def simple_plan(crop):
    return safe_gemini(
        f"Give practical farming plan for {crop} in simple bullet points.",
        "Use fertilizer, irrigation and pest control."
    )

def risks(crop):
    return safe_gemini(
        f"What risks farmer should know before growing {crop}? short bullet points.",
        "Weather and pest risks possible."
    )

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

crop_dict = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea',
    4: 'coconut', 5: 'coffee', 6: 'cotton', 7: 'grapes',
    8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize',
    12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya',
    18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice',
    21: 'watermelon'
}

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;color:#2E7D32;'>🌾 Smart Crop Advisor AI</h1>
<p style='text-align:center;font-size:18px;'>
AI-Powered Crop Recommendation • Simple • Farmer Friendly
</p>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------
with st.form("form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        N = st.number_input("Nitrogen", 0.0, 300.0, 50.0)
        P = st.number_input("Phosphorus", 0.0, 300.0, 30.0)
        K = st.number_input("Potassium", 0.0, 300.0, 40.0)

    with c2:
        temp = st.number_input("Temperature °C", 0.0, 60.0, 25.0)
        humidity = st.number_input("Humidity %", 0.0, 100.0, 60.0)

    with c3:
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.0)
        rainfall = st.number_input("Rainfall mm", 0.0, 500.0, 100.0)

    submit = st.form_submit_button("🚀 Predict Crop")

# ---------------- OUTPUT ----------------
if submit:
    st.session_state.predicted = True
    st.session_state.last_lang = lang

    X = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    probs = model.predict_proba(X)[0]
    top_idx = probs.argsort()[-3:][::-1]
    top_probs = probs[top_idx]
    norm_probs = top_probs / top_probs.sum()

    # ---------------- TOP CARDS ----------------
    st.subheader("🌱 Top Recommendations")
    cols = st.columns(3)

    for i, idx in enumerate(top_idx):
        crop = crop_dict[idx]
        score = int(norm_probs[i] * 100)

        status = "🟢 Best" if score > 60 else "🟡 Moderate" if score > 30 else "🔴 Risky"

        with cols[i]:
            with st.container(border=True):
                st.subheader(f"{crop.capitalize()}")
                st.progress(score / 100)
                st.caption(f"{score}% Match • {status}")

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🤔 Why This Crop",
        "🧠 Farming Plan",
        "⚠️ Risks"
    ])

    # ---------------- OVERVIEW ----------------
    with tab1:
        df = pd.DataFrame({
            "Crop": [crop_dict[i] for i in top_idx],
            "Score": norm_probs * 100
        })

        fig = px.bar(df, x="Crop", y="Score", text="Score", color="Score")
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"✅ Best Choice: {crop_dict[top_idx[0]].upper()}")

    # ---------------- WHY ----------------
    with tab2:
        with st.spinner("Generating..."):
            for idx in top_idx:
                crop = crop_dict[idx]
                st.markdown(f"### 🌾 {crop.capitalize()}")
                st.write(simple_reason(crop))
                st.divider()

    # ---------------- PLAN ----------------
    with tab3:
        with st.spinner("Generating..."):
            for idx in top_idx:
                crop = crop_dict[idx]
                st.markdown(f"### 🌾 {crop.capitalize()}")
                st.write(simple_plan(crop))
                st.divider()

    # ---------------- RISKS ----------------
    with tab4:
        with st.spinner("Generating..."):
            for idx in top_idx:
                crop = crop_dict[idx]
                st.markdown(f"### 🌾 {crop.capitalize()}")
                st.write(risks(crop))
                st.divider()

    st.caption(f"Generated on {datetime.now().strftime('%d %b %Y')}")