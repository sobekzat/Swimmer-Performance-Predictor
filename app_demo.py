"""
CSS 324 Final Project — Interactive Demo App
Swimmer Performance Predictor
Run: streamlit run app_demo.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os

# ─── Load artefacts (graceful fallback if model not yet trained) ───
@st.cache_resource
def load_model():
    model = joblib.load("best_model_gb.pkl")
    return model

@st.cache_data
def load_nation_freq():
    with open("nation_freq.json") as f:
        return json.load(f)

# ─── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Swimmer Performance Predictor", page_icon="🏊", layout="centered")

st.title("🏊 Swimmer Performance Predictor")
st.caption("CSS 324 · Final Project Demo · Gradient Boosting Model")

st.markdown("""
Predict a competitive swimmer's **pace (sec/100m)** and estimated **race time**
based on physical characteristics and race distance.
""")

# ─── Sidebar: About ───────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.info("""
**Dataset**: swim_simulated_performance.csv  
**Records**: 2,787 swimmers  
**Target**: Pace (sec/100m)  
**Final model**: Gradient Boosting  
**R²**: ~0.78 · **MAE**: ~2.5 sec/100m
""")
    st.header("Feature Descriptions")
    st.markdown("""
- **Height / Weight** — physical size
- **BMI** — auto-computed from above
- **Age** — swimmer's age in 2024
- **Sex** — biological sex
- **Distance** — race distance (m)
- **Nationality** — used as frequency encoding
""")

# ─── Input form ───────────────────────────────────────────────────
st.subheader("Enter Swimmer Details")

col1, col2 = st.columns(2)
with col1:
    height = st.slider("Height (m)", 1.43, 2.21, 1.80, 0.01)
    weight = st.slider("Weight (kg)", 45.0, 120.0, 75.0, 0.5)
    age    = st.slider("Age", 22, 49, 26)

with col2:
    sex      = st.radio("Sex", ["male", "female"], horizontal=True)
    distance = st.selectbox("Race Distance (m)", [200, 400, 800, 1500])
    nationality = st.selectbox("Nationality (IOC code)", 
                               ['AUS','BRA','USA','ITA','CHN','JPN','RUS','HUN',
                                'ESP','GBR','FRA','GER','CAN','RSA','NED',
                                'POL','SWE','DEN','NOR','ARG','OTHER'])

predict_btn = st.button("🔮 Predict Performance", use_container_width=True, type="primary")

# ─── Prediction ───────────────────────────────────────────────────
if predict_btn:
    try:
        model = load_model()
        nation_freq = load_nation_freq()

        bmi         = weight / (height ** 2)
        hw_ratio    = height / weight
        sex_binary  = 1 if sex == "male" else 0
        log_dist    = np.log(distance)
        nat_freq    = nation_freq.get(nationality, 10)  # default 10 if unknown

        features = np.array([[height, weight, age, bmi, hw_ratio,
                               sex_binary, log_dist, nat_freq]])
        
        pace_pred = model.predict(features)[0]
        time_pred = pace_pred * distance / 100

        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pace", f"{pace_pred:.2f} sec/100m")
        c2.metric("Race Time", f"{time_pred:.2f} sec")

        mins = int(time_pred // 60)
        secs = time_pred % 60
        c3.metric("Race Time (min:sec)", f"{mins}:{secs:05.2f}")

        # Context gauge: compare to dataset mean pace
        mean_pace = 61.38
        diff = pace_pred - mean_pace
        if diff < -2:
            rating = "🟢 Faster than average"
        elif diff < 2:
            rating = "🟡 Average performer"
        else:
            rating = "🔴 Below average"

        st.markdown(f"**Performance rating:** {rating}  \n"
                    f"*(Dataset mean pace: {mean_pace:.2f} sec/100m)*")

        # Input summary
        with st.expander("Input features used"):
            st.json({
                "height_m": height, "weight_kg": weight, "age": age,
                "sex": sex, "distance_m": distance, "nationality": nationality,
                "bmi": round(bmi, 2), "hw_ratio": round(hw_ratio, 4),
                "log_distance": round(log_dist, 4), "nation_freq": nat_freq
            })

    except FileNotFoundError:
        st.error("Model not found. Please run `swim_performance_project.py` first to train and save the model.")

# ─── Batch prediction ─────────────────────────────────────────────
st.markdown("---")
with st.expander("📁 Batch Prediction (upload CSV)"):
    st.markdown("""Upload a CSV with columns:  
`height, weight, age_2024, sex, distance_m, nationality`""")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            model = load_model()
            nation_freq = load_nation_freq()
            batch = pd.read_csv(uploaded)
            batch['bmi']        = batch['weight'] / (batch['height'] ** 2)
            batch['hw_ratio']   = batch['height'] / batch['weight']
            batch['sex_binary'] = (batch['sex'].str.lower() == 'male').astype(int)
            batch['log_distance'] = np.log(batch['distance_m'])
            batch['nation_freq']  = batch['nationality'].map(nation_freq).fillna(10)

            FEATURE_COLS = ['height','weight','age_2024','bmi','hw_ratio',
                            'sex_binary','log_distance','nation_freq']
            batch['predicted_pace'] = model.predict(batch[FEATURE_COLS])
            batch['predicted_time'] = batch['predicted_pace'] * batch['distance_m'] / 100
            st.dataframe(batch[['name' if 'name' in batch.columns else batch.columns[0],
                                  'distance_m','predicted_pace','predicted_time']].head(20))
            st.download_button("⬇ Download results", batch.to_csv(index=False),
                               "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")
