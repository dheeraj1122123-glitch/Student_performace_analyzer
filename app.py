import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Page Config ---
st.set_page_config(page_title="Student Analytics Pro", layout="wide")

# --- Custom CSS for Dark Theme & Animation ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #0e1117; color: #4caf50; text-align: center; padding: 10px; font-weight: bold; border-top: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- Model Training (Cached) ---
@st.cache_resource
def train_model():
    df = pd.read_csv('student-mat.csv', sep=';')
    features = ['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'internet', 'health', 'G1', 'G2']
    X = df[features].copy()
    le = LabelEncoder()
    X['internet'] = le.fit_transform(X['internet'])
    y_reg, y_clf = df['G3'], df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_reg)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_clf)
    return reg, clf, le

reg_model, clf_model, le = train_model()

# --- Sidebar / Header ---
st.title("🎓 AI Student Performance Dashboard")
st.markdown("Enter details to see academic analysis whether you will pass or not")

# --- Layout: Columns for Inputs ---
with st.sidebar:
    st.header("⚙️ Set your detail")
    studytime = st.selectbox("Study Time", [1, 2, 3, 4], format_func=lambda x: ["<2 hrs", "2-5 hrs", "5-10 hrs", ">10 hrs"][x-1])
    failures = st.slider("Past Failures", 0, 3, 0)
    absences = st.number_input("Absences", 0, 93, 5)
    health = st.select_slider("Health Rating", options=[1, 2, 3, 4, 5], value=3)
    internet = st.radio("Home Internet?", ["yes", "no"], horizontal=True)
    st.divider()
    g1 = st.slider("G1 Marks (0-20)", 0, 20, 10)
    g2 = st.slider("G2 Marks (0-20)", 0, 20, 11)
    medu = st.sidebar.hidden = 3 # Default intermediate
    fedu = st.sidebar.hidden = 3

# --- Main Dashboard ---
if st.button("🚀 Tap to see result"):
    # Prediction
    internet_enc = 1 if internet == "yes" else 0
    # Fixed input array to match model features: 
    # ['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'internet', 'health', 'G1', 'G2']
    input_data = np.array([[studytime, failures, absences, 3, 3, internet_enc, health, g1, g2]])
    
    grade = reg_model.predict(input_data)[0]
    status = clf_model.predict(input_data)[0]
    percentage = (grade / 20) * 100

    # Layout for Results
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Final Grade", f"{grade:.1f}/20")
    m2.metric("Percentage", f"{percentage:.2f}%")
    m3.metric("Status", "PASS ✅" if status == 1 else "FAIL ❌")

    st.divider()

    # --- GRAPHS SECTION ---
    col_left, col_right = st.columns(2)

    with col_left:
        # 1. Gauge Chart (Meter)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = percentage,
            title = {'text': "Academic Performance Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4caf50"},
                'steps': [
                    {'range': [0, 40], 'color': "#ff4b4b"},
                    {'range': [40, 70], 'color': "#ffa500"},
                    {'range': [70, 100], 'color': "#2e7d32"}],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        # 2. Comparison Bar Chart (G1 vs G2 vs Predicted G3)
        comparison_data = pd.DataFrame({
            'Exams': ['G1 (Past)', 'G2 (Midterm)', 'G3 (Predicted)'],
            'Marks': [g1, g2, grade]
        })
        fig_bar = px.bar(comparison_data, x='Exams', y='Marks', color='Marks', 
                         title="Grade Progression Analysis", color_continuous_scale='Greens')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_bar, use_container_width=True)

    if status == 1: st.balloons()
else:
    st.info("👈 Adjust values in the sidebar and click 'Tap to see result' to see results.")

# --- Footer ---
st.markdown(f"""
    <div class="footer">
        Academic performance Analysis Engine | Developed by [Dheeraj Prajapat]
    </div>
    """, unsafe_allow_html=True)