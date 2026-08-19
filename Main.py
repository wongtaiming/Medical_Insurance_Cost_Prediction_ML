import streamlit as st
import pandas as pd
import joblib

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

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Main content */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #2563eb;
    }

    /* Main headings */
    h1 {
        color: #111827;
        font-weight: 700;
    }

    h2 {
        color: #1f2937;
        font-weight: 650;
    }

    h3 {
        color: #374151;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 0.7rem 1rem;
        font-weight: 600;
        background-color: #2563eb;
        color: white;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
    }

    /* Input boxes */
    div[data-baseweb="input"] {
        border-radius: 8px;
    }

    div[data-baseweb="select"] {
        border-radius: 8px;
    }

    /* Cards */
    .info-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(
            135deg,
            #2563eb,
            #3b82f6
        );
        padding: 45px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #e0ecff;
        font-size: 18px;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        text-align: center;
        min-height: 150px;
    }

    .feature-icon {
        font-size: 35px;
        margin-bottom: 10px;
    }

    /* Prediction result */
    .prediction-card {
        background: white;
        padding: 35px;
        border-radius: 18px;
        border: 2px solid #22c55e;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 5px 20px rgba(34,197,94,0.12);
    }

    .prediction-title {
        font-size: 18px;
        color: #6b7280;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 700;
        color: #16a34a;
        margin: 10px 0;
    }

    .prediction-model {
        color: #6b7280;
        font-size: 15px;
    }

    /* Section title */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Disclaimer */
    .disclaimer {
        background-color: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 15px;
        border-radius: 8px;
        color: #7c2d12;
        margin-top: 25px;
    }

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Model paths (update if your filenames differ)
# ----------------------------
MODEL_PATHS = {
    "Random Forest": "models/random_forest_model.pkl",
    "Multiple Linear Regression": "models/linear_regression_model.pkl",
    "SVR (RBF Kernel)": "models/svr_rbf_model.pkl",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.markdown("""
<div style="
    text-align:center;
    padding:20px 5px;
">
    <div style="font-size:45px;">🏥</div>
    <h2 style="color:#2563eb;">
        Medical Insurance AI
    </h2>
    <p style="color:#6b7280;">
        Machine Learning Predictor
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🧮 Calculator"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Educational Machine Learning Project"
)
page = st.sidebar.radio("Go to", ["🏠 Home", "🧮 Calculator"])

# ----------------------------
# HOME PAGE
# ----------------------------
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>🏥 Medical Insurance AI</h1>
        <p>
            Predict your estimated annual medical insurance cost
            using machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">
        🔍 What can this application do?
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3>3 ML Models</h3>
            <p>
                Compare Random Forest, Linear Regression
                and SVR predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Data Driven</h3>
            <p>
                Predictions are generated from
                machine learning models.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Instant Results</h3>
            <p>
                Enter your information and receive
                an estimated cost immediately.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">

    ### 🧠 Available Prediction Models

    **🌲 Random Forest**  
    Uses multiple decision trees to capture complex
    relationships between the input variables.

    **📈 Multiple Linear Regression**  
    Provides a simple and interpretable baseline
    for predicting insurance charges.

    **〰️ SVR (RBF Kernel)**  
    Uses a non-linear support vector approach to
    model more complex relationships.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">

    ### 🧭 How to use

    **1.** Go to **Calculator** using the sidebar.

    **2.** Select your machine learning model.

    **3.** Enter your personal, health and lifestyle information.

    **4.** Click **Predict**.

    **5.** View your estimated annual insurance cost.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b>
        This application is for educational and illustrative
        purposes only. Predictions should not be considered
        an actual insurance quotation.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# CALCULATOR PAGE
# ----------------------------
elif page == "🧮 Calculator":

    st.title("🧮 Insurance Cost Calculator")

    st.write(
        "Enter your information below to estimate your annual "
        "medical insurance cost."
    )

    # -----------------------------
    # Model Selection
    # -----------------------------

    st.markdown(
        '<div class="section-title">🤖 Prediction Model</div>',
        unsafe_allow_html=True
    )

    model_choice = st.selectbox(
        "Select Prediction Model",
        list(MODEL_PATHS.keys())
    )

    st.divider()

    # -----------------------------
    # Personal Information
    # -----------------------------

    st.markdown(
        '<div class="section-title">👤 Personal Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

    # -----------------------------
    # Health Information
    # -----------------------------

    st.markdown(
        '<div class="section-title">❤️ Health Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

    with col2:
        children = st.number_input(
            "Children / Dependents",
            min_value=0,
            max_value=10,
            value=0
        )

    # -----------------------------
    # Lifestyle Information
    # -----------------------------

    st.markdown(
        '<div class="section-title">🚬 Lifestyle & Location</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        smoker = st.selectbox(
            "Smoking Status",
            ["Yes", "No"]
        )

    with col2:
        region = st.selectbox(
            "Region",
            [
                "Northeast",
                "Northwest",
                "Southeast",
                "Southwest"
            ]
        )

    st.divider()

    # -----------------------------
    # Prediction
    # -----------------------------

    if st.button("🔮 Predict Insurance Cost", type="primary"):

        try:
            model = load_model(MODEL_PATHS[model_choice])

        except FileNotFoundError:

            st.error(
                f"Model file not found for **{model_choice}**."
            )

        else:

            input_data = pd.DataFrame({
                "cat__sex_male": [
                    1 if sex == "Male" else 0
                ],

                "cat__smoker_yes": [
                    1 if smoker == "Yes" else 0
                ],

                "cat__region_northwest": [
                    1 if region == "Northwest" else 0
                ],

                "cat__region_southeast": [
                    1 if region == "Southeast" else 0
                ],

                "cat__region_southwest": [
                    1 if region == "Southwest" else 0
                ],

                "remainder__age": [age],

                "remainder__bmi": [bmi],

                "remainder__children": [children]
            })

            prediction = model.predict(input_data)

            estimated_cost = prediction[0]

            st.markdown(f"""
            <div class="prediction-card">

                <div class="prediction-title">
                    Estimated Annual Insurance Cost
                </div>

                <div class="prediction-value">
                    ${estimated_cost:,.2f}
                </div>

                <div class="prediction-model">
                    Prediction generated using {model_choice}
                </div>

            </div>
            """, unsafe_allow_html=True)

            st.info(
                "💡 This is an estimated prediction based on "
                "the information you provided."
            )