import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --navy: #1B2A41;
        --navy-light: #2C4266;
        --paper: #F5F2EA;
        --paper-card: #FCFAF5;
        --ink: #23272E;
        --ink-soft: #5B6472;
        --teal: #2F6F5E;
        --teal-soft: #E7F0EC;
        --line: #DCD5C4;
        --amber: #9C6B34;
        --amber-soft: #F5EDE0;
    }

    /* Main background */
    .stApp {
        background-color: var(--paper);
    }

    body, .stApp, p, span, div, label {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }

    /* Main content */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--navy);
        border-right: none;
    }

    section[data-testid="stSidebar"] * {
        color: #E7ECF3 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Source Serif 4', serif;
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Source Serif 4', serif;
        color: var(--navy);
        letter-spacing: -0.01em;
    }

    h1 { font-weight: 700; }
    h2 { font-weight: 600; }
    h3 { font-weight: 600; color: var(--ink); }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        border: 1px solid var(--navy);
        padding: 0.7rem 1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        background-color: var(--navy);
        color: #ffffff;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        background-color: var(--navy-light);
        border-color: var(--navy-light);
    }

    /* Input boxes */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        border-radius: 4px;
    }

    /* Cards */
    .info-card {
        background-color: var(--paper-card);
        padding: 26px 28px;
        border-radius: 8px;
        border: 1px solid var(--line);
        margin-bottom: 20px;
    }

    /* Hero section */
    .hero {
        background: var(--navy);
        padding: 40px 44px;
        border-radius: 10px;
        color: white;
        margin-bottom: 26px;
        border-left: 5px solid var(--teal);
    }

    .hero-eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 12px;
        color: #9FB3CC;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        margin: 0 0 12px 0;
    }

    .hero p {
        color: #D6DFEA;
        font-size: 16px;
        line-height: 1.55;
        max-width: 560px;
    }

    .hero-stat-row {
        display: flex;
        gap: 32px;
        margin-top: 22px;
        border-top: 1px solid rgba(255,255,255,0.15);
        padding-top: 18px;
    }

    .hero-stat-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 22px;
        font-weight: 600;
        color: #ffffff;
    }

    .hero-stat-label {
        font-size: 12.5px;
        color: #9FB3CC;
        margin-top: 2px;
    }

    /* Model cards on home page */
    .model-card {
        background: var(--paper-card);
        padding: 20px 22px;
        border-radius: 8px;
        border: 1px solid var(--line);
        border-left: 3px solid var(--teal);
        height: 100%;
    }

    .model-card h4 {
        font-family: 'Source Serif 4', serif;
        margin: 0 0 6px 0;
        color: var(--navy);
        font-size: 17px;
    }

    .model-card p {
        color: var(--ink-soft);
        font-size: 14px;
        line-height: 1.5;
        margin: 0;
    }

    /* Step tracker */
    .step-tracker {
        display: flex;
        justify-content: space-between;
        margin: 8px 0 28px 0;
        position: relative;
    }

    .step-tracker::before {
        content: "";
        position: absolute;
        top: 13px;
        left: 5%;
        right: 5%;
        height: 1px;
        background: var(--line);
        z-index: 0;
    }

    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        z-index: 1;
        background: var(--paper);
        padding: 0 10px;
    }

    .step-num {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        border: 1.5px solid var(--navy);
        color: var(--navy);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        background: var(--paper-card);
    }

    .step-label {
        font-size: 12px;
        color: var(--ink-soft);
        margin-top: 6px;
        font-weight: 500;
        white-space: nowrap;
    }

    /* Section title */
    .section-title {
        font-family: 'Source Serif 4', serif;
        font-size: 20px;
        font-weight: 600;
        color: var(--navy);
        margin-top: 8px;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--line);
    }

    /* Prediction result */
    .prediction-card {
        background: var(--paper-card);
        padding: 32px 34px;
        border-radius: 10px;
        border: 1px solid var(--line);
        border-top: 4px solid var(--teal);
        margin-top: 22px;
    }

    .prediction-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--ink-soft);
        font-weight: 600;
    }

    .prediction-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 40px;
        font-weight: 600;
        color: var(--navy);
        margin: 8px 0 4px 0;
    }

    .prediction-model {
        color: var(--ink-soft);
        font-size: 13.5px;
        margin-bottom: 20px;
    }

    /* Range gauge */
    .range-track {
        position: relative;
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--teal-soft), var(--teal) 50%, var(--amber-soft));
        margin: 18px 0 8px 0;
    }

    .range-marker {
        position: absolute;
        top: -7px;
        width: 3px;
        height: 22px;
        background: var(--navy);
        border-radius: 2px;
    }

    .range-labels {
        display: flex;
        justify-content: space-between;
        font-size: 11.5px;
        color: var(--ink-soft);
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Disclaimer */
    .disclaimer {
        background-color: var(--amber-soft);
        border-left: 4px solid var(--amber);
        padding: 14px 18px;
        border-radius: 6px;
        color: #5C3E1A;
        margin-top: 22px;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Model paths (update if your filenames differ)
# ----------------------------
MODEL_PATHS = {
    "Random Forest": "models/random_forest_model.pkl",
    "Multiple Linear Regression": "models/MLR_model.pkl",
    "SVR (RBF Kernel)": "models/svr_rbf_model.pkl",
}

SCALER_PATHS = {
    "SVR (RBF Kernel)": "models/svr_scaler.pkl",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_scaler(path):
    return joblib.load(path)

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.markdown("""
<div style="text-align:center; padding:22px 5px 10px 5px;">
    <div style="font-size:32px;">🏥</div>
    <h2 style="margin:6px 0 2px 0;">Medical Insurance AI</h2>
    <p style="color:#9FB3CC; font-size:13px;">Machine Learning Predictor</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("Educational Machine Learning Project")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧮 Calculator"])

# ----------------------------
# HOME PAGE
# ----------------------------
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Educational ML Project</div>
        <h1>Medical Insurance Cost Predictor</h1>
        <p>
            Three machine learning models, trained on the same health and
            lifestyle data, estimate an annual insurance charge — so you
            can see how different modelling approaches reach different answers.
        </p>
        <div class="hero-stat-row">
            <div>
                <div class="hero-stat-num">3</div>
                <div class="hero-stat-label">Models compared</div>
            </div>
            <div>
                <div class="hero-stat-num">7</div>
                <div class="hero-stat-label">Input features</div>
            </div>
            <div>
                <div class="hero-stat-num">&lt;1s</div>
                <div class="hero-stat-label">Prediction time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Available prediction models</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="model-card">
            <h4>🌲 Random Forest</h4>
            <p>Combines many decision trees to capture non-linear
            relationships between age, health, and lifestyle factors.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="model-card">
            <h4>📈 Linear Regression</h4>
            <p>A transparent baseline — each feature has a fixed,
            interpretable effect on the predicted cost.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="model-card">
            <h4>〰️ SVR (RBF Kernel)</h4>
            <p>Fits a flexible, non-linear boundary through the
            data for potentially sharper predictions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="margin-top:20px;">

    #### How to use this tool

    Go to **Calculator** in the sidebar, choose a model, enter your
    personal, health, and lifestyle details, then select **Predict**
    to see an estimated annual cost.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This application is for educational and
        illustrative purposes only. Predictions should not be considered
        an actual insurance quotation.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# CALCULATOR PAGE
# ----------------------------
elif page == "🧮 Calculator":

    st.title("Insurance Cost Calculator")
    st.write("Enter your information below to estimate your annual medical insurance cost.")

    st.markdown("""
    <div class="step-tracker">
        <div class="step"><div class="step-num">1</div><div class="step-label">Model</div></div>
        <div class="step"><div class="step-num">2</div><div class="step-label">Personal</div></div>
        <div class="step"><div class="step-num">3</div><div class="step-label">Health</div></div>
        <div class="step"><div class="step-num">4</div><div class="step-label">Lifestyle</div></div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Model Selection
    # -----------------------------

    st.markdown('<div class="section-title">🤖 Prediction Model</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Prediction Model",
        list(MODEL_PATHS.keys())
    )

    # -----------------------------
    # Personal Information
    # -----------------------------

    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])

    # -----------------------------
    # Health Information
    # -----------------------------

    st.markdown('<div class="section-title">❤️ Health Information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    with col2:
        children = st.number_input("Children / Dependents", min_value=0, max_value=10, value=0)

    # -----------------------------
    # Lifestyle Information
    # -----------------------------

    st.markdown('<div class="section-title">🚬 Lifestyle & Location</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        smoker = st.selectbox("Smoking Status", ["Yes", "No"])

    with col2:
        region = st.selectbox(
            "Region",
            ["Northeast", "Northwest", "Southeast", "Southwest"]
        )

    st.write("")

    # -----------------------------
    # Prediction
    # -----------------------------

    if st.button("🔮 Predict Insurance Cost", type="primary"):

        try:
            model = load_model(MODEL_PATHS[model_choice])

        except FileNotFoundError:
            st.error(f"Model file not found for **{model_choice}**.")

        else:
            smoker_flag = 1 if smoker == "Yes" else 0

            input_data = pd.DataFrame({
                "cat__sex_male": [1 if sex == "Male" else 0],
                "cat__smoker_yes": [smoker_flag],
                "cat__region_northwest": [1 if region == "Northwest" else 0],
                "cat__region_southeast": [1 if region == "Southeast" else 0],
                "cat__region_southwest": [1 if region == "Southwest" else 0],
                "remainder__age": [age],
                "remainder__bmi": [bmi],
                "remainder__children": [children],
                "remainder__smoker_bmi": [smoker_flag * bmi],
                "remainder__age_squared": [age ** 2],
                "remainder__age_smoker": [age * smoker_flag],
                "remainder__is_underweight": [int(bmi < 18.5)], 
                "remainder__is_obese": [int(bmi >= 30)], 
                "remainder__obese_smoker": [int(bmi >= 30) * smoker_flag],  
                "remainder__children_smoker": [children * smoker_flag],
            })

            # Apply scaling only if this model needs it (SVR)
            if model_choice in SCALER_PATHS:
                scaler = load_scaler(SCALER_PATHS[model_choice])
                input_for_model = scaler.transform(input_data)
            else:
                input_for_model = input_data

            prediction = model.predict(input_for_model)

            # Model was trained on log1p(charges) — invert back to dollars
            LOG1P_TARGET_MODELS = ["Random Forest", "Multiple Linear Regression"]

            if model_choice in LOG1P_TARGET_MODELS:
                estimated_cost = np.expm1(prediction[0])
            else:
                estimated_cost = prediction[0]

            # Purely visual reference band for the gauge below (not a
            # statistical claim — just gives the number some context).
            gauge_min, gauge_max = 1000, 50000
            gauge_pct = min(max((estimated_cost - gauge_min) / (gauge_max - gauge_min), 0), 1) * 100

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">Estimated Annual Insurance Cost</div>
                <div class="prediction-value">${estimated_cost:,.2f}</div>
                <div class="prediction-model">Prediction generated using {model_choice}</div>
                <div class="range-track">
                    <div class="range-marker" style="left:{gauge_pct:.1f}%;"></div>
                </div>
                <div class="range-labels">
                    <span>${gauge_min:,}</span>
                    <span>${gauge_max:,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info(
                "💡 This is an estimated prediction based on the information "
                "you provided, shown against an illustrative $1,000–$50,000 scale."
            )