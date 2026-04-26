import numpy as np
import pickle
import os
import json
import re
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Smart Crop Advisor AI 🌾", layout="wide")

if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "last_lang" not in st.session_state:
    st.session_state.last_lang = "English"
if "show_lang_alert" not in st.session_state:
    st.session_state.show_lang_alert = False

load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-flash-lite-latest")
else:
    gemini_model = None

st.markdown("<h1 style='text-align:center;color:white;'>🌾 Smart Crop Advisor AI</h1>", unsafe_allow_html=True)

sp1, sp2, lang_col = st.columns([6,1,1])

with lang_col:
    lang = st.selectbox("", ["English","Hindi","Marathi"], label_visibility="collapsed")

lang_map = {
    "English": "Respond in English",
    "Hindi": "Respond in Hindi",
    "Marathi": "Respond in Marathi"
}

if st.session_state.predicted and lang != st.session_state.last_lang:
    if not st.session_state.show_lang_alert:
        st.toast("⚠️ Please click Predict again to apply language change")
        st.session_state.show_lang_alert = True

def clean_name(name):
    name = name.strip()
    name = re.sub(r"\(.*?\)", "", name)
    return name.strip()

with open("states-and-districts.json") as f:
    data = json.load(f)

state_map = {
    item["state"]: [clean_name(d) for d in item["districts"]]
    for item in data["states"]
}

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={WEATHER_KEY}&units=metric"
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
    except:
        return None

def get_rainfall(weather):
    if not weather:
        return None
    return weather.get("rain", {}).get("1h") or weather.get("rain", {}).get("3h") or 0

def rainfall_status(rain):
    if rain == 0:
        return "☀️ No rain"
    elif rain < 2:
        return "🌦 Light rain"
    elif rain < 10:
        return "🌧 Moderate rain"
    else:
        return "⛈ Heavy rain"

def safe_gemini(prompt, fallback):
    try:
        if gemini_model is None:
            return fallback
        res = gemini_model.generate_content(prompt + f"\n{lang_map[lang]}")
        return res.text if res.text else fallback
    except:
        return fallback

def why_prompt(crop):
    return f"Crop: {crop}\nExplain clearly in bullet points with emojis."

def plan_prompt(crop):
    return f"Crop: {crop}\nGive step-by-step farming plan in bullets."

def risk_prompt(crop):
    return f"Crop: {crop}\nExplain risks in bullet points."

def crop_weather_advice(crop, weather, rainfall):
    return safe_gemini(
        f"""
        Crop: {crop}
        Temp: {weather['main']['temp']}°C
        Humidity: {weather['main']['humidity']}%
        Rainfall: {rainfall} mm
        Condition: {weather['weather'][0]['description']}
        Give short bullet farmer advice.
        """,
        "No advice"
    )

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

c1, c2 = st.columns(2)

with c1:
    state = st.selectbox("State", list(state_map.keys()), index=list(state_map.keys()).index("Maharashtra"))

with c2:
    district = st.selectbox("District", state_map[state], index=state_map[state].index("Pune") if "Pune" in state_map[state] else 0)

weather = get_weather(district)
api_rain = get_rainfall(weather) if weather else None

if weather:
    st.info(
        f"🌦 Temp: {weather['main']['temp']}°C | "
        f"💧 Humidity: {weather['main']['humidity']}% | "
        f"{rainfall_status(api_rain)}"
    )

with st.form("form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        N = st.number_input("Nitrogen", 0.0, 300.0, 50.0)
        P = st.number_input("Phosphorus", 0.0, 300.0, 30.0)
        K = st.number_input("Potassium", 0.0, 300.0, 40.0)

    with c2:
        temp = st.number_input("Temperature °C", 0.0, 60.0, float(weather['main']['temp']) if weather else 25.0)
        humidity = st.number_input("Humidity %", 0.0, 100.0, float(weather['main']['humidity']) if weather else 60.0)

    with c3:
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.0)
        rainfall = st.number_input("Rainfall mm", 0.0, 500.0, float(api_rain) if api_rain is not None else 100.0)

    submit = st.form_submit_button("🚀 Predict Crop")

if submit:
    st.session_state.predicted = True
    st.session_state.last_lang = lang
    st.session_state.show_lang_alert = False

    X = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    probs = model.predict_proba(X)[0]
    top_idx = probs.argsort()[-3:][::-1]
    norm_probs = probs[top_idx] / probs[top_idx].sum()

    scores = [round(p * 100) for p in norm_probs]
    scores[0] += 100 - sum(scores)

    st.subheader("🌱 Top Recommendations")
    cols = st.columns(3)

    for i, idx in enumerate(top_idx):
        crop = crop_dict[idx]
        score = scores[i]
        status = "🟢 Best" if score > 60 else "🟡 Moderate" if score > 30 else "🔴 Risky"

        with cols[i]:
            with st.container(border=True):
                st.subheader(crop.capitalize())
                st.progress(score / 100)
                st.caption(f"{score}% • {status}")

    tabs = st.tabs(["📊 Overview","🤔 Why","🧠 Plan","⚠️ Risks"] + (["🌦 Weather Impact"] if weather else []))

    with tabs[0]:
        df = pd.DataFrame({
            "Crop": [crop_dict[i] for i in top_idx],
            "Score": scores
        })
        st.plotly_chart(px.bar(df, x="Crop", y="Score", text="Score"), use_container_width=True)
        st.success(f"🌾 Suggested Crop: {crop_dict[top_idx[0]].upper()}")

    with tabs[1]:
        for idx in top_idx:
            crop = crop_dict[idx]
            with st.container(border=True):
                st.markdown(f"### 🌾 {crop.capitalize()}")
                with st.spinner("Generating..."):
                    st.markdown(safe_gemini(why_prompt(crop), ""))

    with tabs[2]:
        for idx in top_idx:
            crop = crop_dict[idx]
            with st.container(border=True):
                st.markdown(f"### 🌾 {crop.capitalize()}")
                with st.spinner("Generating..."):
                    st.markdown(safe_gemini(plan_prompt(crop), ""))

    with tabs[3]:
        for idx in top_idx:
            crop = crop_dict[idx]
            with st.container(border=True):
                st.markdown(f"### 🌾 {crop.capitalize()}")
                with st.spinner("Generating..."):
                    st.markdown(safe_gemini(risk_prompt(crop), ""))

    if weather:
        with tabs[4]:
            for idx in top_idx:
                crop = crop_dict[idx]
                with st.container(border=True):
                    st.markdown(f"### 🌾 {crop.capitalize()}")
                    with st.spinner("Generating..."):
                        st.markdown(crop_weather_advice(crop, weather, rainfall))

    st.caption(f"Generated on {datetime.now().strftime('%d %b %Y')}")