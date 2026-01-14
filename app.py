import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from streamlit_lottie import st_lottie
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# --- Page Config ---
st.set_page_config(page_title="Student Final Result Check", layout="wide")

# --- Function to load Lottie Animations ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

# 2. Variable define karein (Important)
lottie_main = load_lottieurl("https://lottie.host/80998f6d-7f41-4770-8e11-855799988220/8hG0YIun8T.json") # Study animation

# --- Custom CSS ---
st.markdown("""
    <style>
    /* 1. Puri Screen ka Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    

    /* 2. Main Title: Golden Floating & Glow Effect */
    .main-title {
        font-size: 50px;
        font-weight: 900;
        background: linear-gradient(to bottom, #FFD700 22%, #FF8C00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.5));
        animation: float 4s ease-in-out infinite;
        text-align: center;
    }

    /* 3. 'Presented by' Shining Animation */
    .presented-by {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        background: linear-gradient(90deg, #B8860B, #FFD700, #B8860B);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 2s linear infinite;
    }

    /* 4. Golden Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #BF953F, #FCF6BA, #B38728, #FBF5B7, #AA771C);
        color: black !important;
        font-weight: bold;
        border: none;
        border-radius: 30px;
        transition: 0.5s;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(255, 215, 0, 0.7);
    }

    /* 5. Metrics & Cards Decoration */
    [data-testid="stMetricValue"] {
        color: #FFD700 !important;
    }
    .stMetric {
        background: rgba(43, 33, 1, 0.6);
        border: 1px solid #BF953F;
        border-radius: 15px;
    }

    /* Animations Keyframes */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
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

# --- Header Section ---

col_t1, col_t2 = st.columns([2, 1])
with col_t1:
    # Presented by with Gradient Shine
    st.markdown('<p class="presented-by">✨ Presented by Dheeraj</p>', unsafe_allow_html=True)
    
    # Main Title with Floating Effect
    st.markdown('<h1 class="main-title" >🎓 AI Student Final Result Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("<p style='color: yellow; font-size: 18px;'>🏆🏆Predict your future academic results by this🏆🏆</p>", unsafe_allow_html=True)
with col_t2:
    if lottie_main:
        st_lottie(lottie_main, height=180, key="main_anim")

st.divider()

# --- Input Section (Main Screen) ---
st.subheader("📝 Enter Details Below")

# Inputs in Grid Layout
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        studytime = st.selectbox("Study Time (Weekly)", [1, 2, 3, 4], format_func=lambda x: ["<2 hrs", "2-5 hrs", "5-10 hrs", ">10 hrs"][x-1])
        health = st.select_slider("Current Health Rating", options=[1, 2, 3, 4, 5], value=3)
    with c2:
        failures = st.slider("Number of Past Failures", 0, 3, 0)
        internet = st.radio("Do you have Home Internet?", ["yes", "no"], horizontal=True)
    with c3:
        absences = st.number_input("Total School Absences", 0, 93, 5)
        st.info("💡 Pro Tip: Consistency is key to success!")

    st.markdown("### 📊 Exam Marks")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        g1 = st.slider("exam1 Marks (terminal/sem)", 0, 20, 10)
    with g_col2:
        g2 = st.slider("exam2 Marks (halfyearly/sem)", 0, 20, 11)

st.write("\n")
# --- Big Predict Button ---
predict_btn = st.button("🚀 CLICK TO SEE RESULT")

# --- Results ---
if predict_btn:
    internet_enc = 1 if internet == "yes" else 0
    input_data = np.array([[studytime, failures, absences, 3, 3, internet_enc, health, g1, g2]])
    
    grade = reg_model.predict(input_data)[0]
    status = clf_model.predict(input_data)[0]
    percentage = (grade / 20) * 100

    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Final Grade", f"{grade:.1f} / 20")
    res_col2.metric("Percentage", f"{percentage:.2f}%")
    res_col3.metric("Final Status", "PASS ✅" if status == 1 else "FAIL ❌")

    # Visualizations
    v1, v2 = st.columns(2)
    with v1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = percentage,
            title = {'text': "Confidence Score"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#4caf50"}}
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with v2:
        fig_bar = px.bar(x=['G1', 'G2', 'Predicted G3'], y=[g1, g2, grade], 
                         labels={'x': 'Exams', 'y': 'Marks'}, title="Academic Trend")
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    if status == 1: st.balloons()
else:
    st.write("---")
    st.markdown("<p style='text-align: center; color: yellow;'>Enter your datails and click the green button above to see result.</p>", unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer" color: yellow;>Academic Performance Model | Developed by Dheeraj Prajapat</div>', unsafe_allow_html=True)
