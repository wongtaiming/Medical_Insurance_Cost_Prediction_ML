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

    .stButton > button p {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: var(--navy-light);
        border-color: var(--navy-light);
    }

    .stButton > button:hover p {
        color: #ffffff !important;
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
        color: #ffffff !important;
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

    /* Explanation section */
    .explain-card {
        background-color: var(--paper-card);
        padding: 22px 26px;
        border-radius: 8px;
        border: 1px solid var(--line);
        border-left: 3px solid var(--amber);
        margin-top: 18px;
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

# Path to the ORIGINAL (unencoded) dataset, used only to build benchmark
# stats for the "Why this estimate?" explanation section. Adjust if needed.
BENCHMARK_DATA_PATH = "dataset/medical_insurance.csv"

# Full engineered feature set used by the models — unchanged from the
# original app, kept here so the explanation functions (linear-regression
# coefficient breakdown / random-forest importance) can label them.
EXPECTED_COLUMNS = [
    "cat__sex_male",
    "cat__smoker_yes",
    "cat__region_northwest",
    "cat__region_southeast",
    "cat__region_southwest",
    "remainder__age",
    "remainder__bmi",
    "remainder__children",
    "remainder__smoker_bmi",
    "remainder__age_squared",
    "remainder__age_smoker",
    "remainder__is_underweight",
    "remainder__is_obese",
    "remainder__obese_smoker",
    "remainder__children_smoker",
]

FEATURE_LABELS = {
    "cat__sex_male": "Sex (Male)",
    "cat__smoker_yes": "Smoking status",
    "cat__region_northwest": "Region (Northwest)",
    "cat__region_southeast": "Region (Southeast)",
    "cat__region_southwest": "Region (Southwest)",
    "remainder__age": "Age",
    "remainder__bmi": "BMI",
    "remainder__children": "Number of children",
    "remainder__smoker_bmi": "Smoker × BMI interaction",
    "remainder__age_squared": "Age² (non-linear age effect)",
    "remainder__age_smoker": "Age × Smoker interaction",
    "remainder__is_underweight": "Underweight flag",
    "remainder__is_obese": "Obese flag",
    "remainder__obese_smoker": "Obese × Smoker interaction",
    "remainder__children_smoker": "Children × Smoker interaction",
}

LOG1P_TARGET_MODELS = ["Random Forest", "Multiple Linear Regression"]


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_resource
def load_scaler(path):
    return joblib.load(path)


@st.cache_data
def load_benchmark_stats():
    """
    Loads the raw (unencoded) dataset to compute reference averages used
    for the 'why is my cost high/low' explanation. Falls back to
    reasonable hardcoded defaults if the file can't be found, so the app
    never crashes because of this optional feature.
    """
    try:
        df = pd.read_csv(BENCHMARK_DATA_PATH)
        df.columns = [c.strip().lower() for c in df.columns]

        stats = {
            "overall_avg": df["charges"].mean(),
            "smoker_avg": df.loc[df["smoker"] == "yes", "charges"].mean(),
            "nonsmoker_avg": df.loc[df["smoker"] == "no", "charges"].mean(),
            "avg_age": df["age"].mean(),
            "avg_bmi": df["bmi"].mean(),
            "avg_children": df["children"].mean(),
            "region_avg": df.groupby("region")["charges"].mean().to_dict(),
            "available": True,
        }
        return stats
    except Exception:
        # Fallback reference values (approximate, based on typical
        # public insurance datasets) so the explanation still works.
        return {
            "overall_avg": 13270.0,
            "smoker_avg": 32050.0,
            "nonsmoker_avg": 8434.0,
            "avg_age": 39.0,
            "avg_bmi": 30.7,
            "avg_children": 1.1,
            "region_avg": {
                "northeast": 13406.0,
                "northwest": 12417.0,
                "southeast": 14735.0,
                "southwest": 12347.0,
            },
            "available": False,
        }


# ----------------------------
# Explanation logic (rule-based, works for every model)
# ----------------------------
def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal weight"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"


def build_rule_based_explanation(age, sex, bmi, children, smoker, region, prediction, stats):
    """
    Works for ANY model. Compares the user's inputs against dataset
    averages to explain, in plain language, why the predicted cost is
    higher or lower than typical.
    """
    points = []
    region_key = region.strip().lower()

    # --- Overall comparison ---
    diff_pct = ((prediction - stats["overall_avg"]) / stats["overall_avg"]) * 100
    if diff_pct > 5:
        points.append(
            f"🔺 Your predicted cost is **{diff_pct:.0f}% higher** than the "
            f"average charge (${stats['overall_avg']:,.0f}) across all policyholders."
        )
    elif diff_pct < -5:
        points.append(
            f"🔻 Your predicted cost is **{abs(diff_pct):.0f}% lower** than the "
            f"average charge (${stats['overall_avg']:,.0f}) across all policyholders."
        )
    else:
        points.append(
            f"➖ Your predicted cost is close to the overall average "
            f"(${stats['overall_avg']:,.0f})."
        )

    # --- Smoking status: usually the single biggest driver ---
    if smoker.strip().lower() == "yes":
        gap = stats["smoker_avg"] - stats["nonsmoker_avg"]
        points.append(
            f"🚬 Smoking is the biggest cost driver. On average, smokers pay about "
            f"\${gap:,.0f} more per year than non-smokers: "
            f"\${stats['smoker_avg']:,.0f} vs \${stats['nonsmoker_avg']:,.0f}"
        )
    else:
        points.append(
            "✅ Being a non-smoker significantly lowers your estimated cost — "
            "smokers pay roughly 3–4x more on average in this dataset."
        )

    # --- BMI ---
    category = bmi_category(bmi)
    if category == "obese":
        points.append(
            f"⚖️ Your BMI ({bmi:.1f}) falls in the **obese** range "
            f"(≥ 30), which is associated with higher medical costs, "
            f"especially when combined with smoking."
        )
    elif category == "overweight":
        points.append(
            f"⚖️ Your BMI ({bmi:.1f}) is in the **overweight** range "
            f"(25–29.9), a mild upward factor on cost."
        )
    else:
        points.append(
            f"⚖️ Your BMI ({bmi:.1f}) is in the **{category}** range, "
            f"which does not add significant extra cost."
        )

    # --- Age ---
    if age > stats["avg_age"] + 10:
        points.append(
            f"🎂 At age {age}, you're notably older than the average "
            f"policyholder (~{stats['avg_age']:.0f}). Costs tend to rise with age."
        )
    elif age < stats["avg_age"] - 10:
        points.append(
            f"🎂 At age {age}, you're younger than the average policyholder "
            f"(~{stats['avg_age']:.0f}), which tends to lower costs."
        )

    # --- Children ---
    if children > stats["avg_children"] + 1:
        points.append(
            f"👨‍👩‍👧‍👦 You have {children} dependents, more than the average "
            f"(~{stats['avg_children']:.1f}), which can modestly raise cost."
        )

    # --- Region ---
    if region_key in stats["region_avg"]:
        region_avg = stats["region_avg"][region_key]
        region_diff = ((region_avg - stats["overall_avg"]) / stats["overall_avg"]) * 100
        if abs(region_diff) > 5:
            direction = "higher" if region_diff > 0 else "lower"
            points.append(
                f"📍 The **{region.title()}** region tends to have "
                f"{direction} average charges (${region_avg:,.0f}) compared "
                f"to the overall average."
            )

    if not stats.get("available", True):
        points.append(
            "_(Note: benchmark averages shown here are approximate defaults, "
            "since the reference dataset wasn't found — real dataset stats "
            "will make this more accurate.)_"
        )

    return points


def linear_regression_breakdown(encoded_row: pd.DataFrame, model):
    """
    Only meaningful for a raw (unscaled) Linear Regression model.
    Returns a dataframe of the exact dollar/log-dollar contribution per
    engineered feature for this specific prediction.
    """
    coefs = model.coef_
    intercept = model.intercept_
    values = encoded_row.iloc[0]

    contributions = []
    for col, coef in zip(EXPECTED_COLUMNS, coefs):
        contribution = coef * values[col]
        contributions.append({
            "Feature": FEATURE_LABELS.get(col, col),
            "Contribution": contribution
        })

    contrib_df = pd.DataFrame(contributions)
    contrib_df.loc[len(contrib_df)] = {
        "Feature": "Base amount (intercept)",
        "Contribution": intercept
    }
    contrib_df["Abs"] = contrib_df["Contribution"].abs()
    contrib_df = contrib_df.sort_values("Abs", ascending=False).drop(columns="Abs")
    return contrib_df


def random_forest_importance(model):
    """
    Global feature importance (not specific to this one prediction, but
    still useful context on what the model relies on overall).
    """
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(c, c) for c in EXPECTED_COLUMNS],
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    return imp_df


def render_explanation(raw_input, encoded_row, prediction_dollars, model_choice, model, stats):
    st.markdown("#### 🧠 Why this estimate?")

    # Rule-based comparison — always shown, works for every model
    points = build_rule_based_explanation(
        age=raw_input["age"],
        sex=raw_input["sex"],
        bmi=raw_input["bmi"],
        children=raw_input["children"],
        smoker=raw_input["smoker"],
        region=raw_input["region"],
        prediction=prediction_dollars,
        stats=stats,
    )
    for p in points:
        st.markdown(f"- {p}")

    # Model-specific deeper insight
    with st.expander("🔬 See model-specific details"):
        if model_choice == "Multiple Linear Regression":
            st.write(
                "Linear Regression assigns a fixed weight to each factor "
                "(on the model's training scale). Here's the breakdown for "
                "your inputs, largest effect first:"
            )
            try:
                contrib_df = linear_regression_breakdown(encoded_row, model)
                st.dataframe(
                    contrib_df.style.format({"Contribution": "{:,.4f}"}),
                    use_container_width=True
                )
                st.caption(
                    "Note: this model predicts log-cost, so contributions "
                    "are on a log scale, not raw dollars."
                )
            except Exception:
                st.info(
                    "Couldn't compute an exact breakdown for this model "
                    "(it may have been trained on scaled inputs)."
                )

        elif model_choice == "Random Forest":
            st.write(
                "Random Forest doesn't give a simple per-prediction formula, "
                "but here's which factors it relies on **most overall**, "
                "based on the whole training dataset:"
            )
            try:
                imp_df = random_forest_importance(model)
                st.bar_chart(imp_df.set_index("Feature"))
            except Exception:
                st.info("Feature importance isn't available for this model file.")

        else:  # SVR (RBF Kernel)
            st.write(
                "SVR with an RBF kernel doesn't produce simple per-feature "
                "weights — it measures similarity to patterns seen in "
                "training data in a transformed space. For this model, the "
                "comparison-based explanation above is the most reliable way "
                "to understand your result."
            )
    st.markdown('</div>', unsafe_allow_html=True)


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
        <h1 style="color: #ffffff !important;">Medical Insurance Cost Predictor</h1>
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
    to see an estimated annual cost — plus a **"Why this estimate?"**
    breakdown explaining what pushed the number up or down.

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

            # -----------------------------
            # Rule-based "Why this estimate?" explanation
            # -----------------------------
            stats = load_benchmark_stats()
            raw_input = {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
            render_explanation(
                raw_input=raw_input,
                encoded_row=input_data,
                prediction_dollars=estimated_cost,
                model_choice=model_choice,
                model=model,
                stats=stats,
            )