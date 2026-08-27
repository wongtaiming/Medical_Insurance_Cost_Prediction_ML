import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

# ----------------------------
# Model / scaler / data paths
# ----------------------------
MODEL_PATHS = {
    "Random Forest": "models/random_forest_model.pkl",
    "Multiple Linear Regression": "models/linear_regression_model.pkl",
    "SVR (RBF Kernel)": "models/svr_rbf_model.pkl",
}

SCALER_PATHS = {
    "SVR (RBF Kernel)": "models/svr_scaler.pkl",
}

# Path to the ORIGINAL (unencoded) dataset, used only to build benchmark stats
# for the explanation section. Adjust this path if your file lives elsewhere.
BENCHMARK_DATA_PATH = "dataset/medical_insurance.csv"

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
    "remainder__smoker_bmi": "Smoker × BMI",
    "remainder__age_squared": "Age²",
    "remainder__age_smoker": "Age × Smoker",
    "remainder__is_underweight": "Underweight",
    "remainder__is_obese": "Obese",
    "remainder__obese_smoker": "Obese × Smoker",
    "remainder__children_smoker": "Children × Smoker",
}

RAW_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
VALID_REGIONS = ["northeast", "northwest", "southeast", "southwest"]


# ----------------------------
# Loaders
# ----------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_resource
def load_scaler(path):
    return joblib.load(path)


@st.cache_data
def load_benchmark_stats():
    """
    Loads the raw dataset to compute reference averages used for the
    'why is my cost high/low' explanation. Falls back to reasonable
    hardcoded defaults if the file can't be found, so the app never crashes.
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
# Preprocessing / prediction
# ----------------------------
def preprocess_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df["smoker"] = df["smoker"].astype(str).str.strip().str.lower()
    df["region"] = df["region"].astype(str).str.strip().str.lower()

    invalid_sex = ~df["sex"].isin(["male", "female"])
    invalid_smoker = ~df["smoker"].isin(["yes", "no"])
    invalid_region = ~df["region"].isin(VALID_REGIONS)

    if invalid_sex.any() or invalid_smoker.any() or invalid_region.any():
        bad_rows = df[invalid_sex | invalid_smoker | invalid_region]
        raise ValueError(
            f"Found {len(bad_rows)} row(s) with invalid 'sex', 'smoker', "
            f"or 'region' values. Problem rows (0-indexed): {list(bad_rows.index)}"
        )

    encoded = pd.DataFrame()

# Categorical features
    encoded["cat__sex_male"] = (df["sex"] == "male").astype(int)
    encoded["cat__smoker_yes"] = (df["smoker"] == "yes").astype(int)
    encoded["cat__region_northwest"] = (df["region"] == "northwest").astype(int)
    encoded["cat__region_southeast"] = (df["region"] == "southeast").astype(int)
    encoded["cat__region_southwest"] = (df["region"] == "southwest").astype(int)

# Original numerical features
    encoded["remainder__age"] = df["age"].astype(float)
    encoded["remainder__bmi"] = df["bmi"].astype(float)
    encoded["remainder__children"] = df["children"].astype(int)

# Get smoker flag
    smoker_flag = (df["smoker"] == "yes").astype(int)

# Feature engineering
    encoded["remainder__smoker_bmi"] = smoker_flag * df["bmi"].astype(float)

    encoded["remainder__age_squared"] = df["age"].astype(float) ** 2

    encoded["remainder__age_smoker"] = (
        df["age"].astype(float) * smoker_flag
    )

    encoded["remainder__is_underweight"] = (
        df["bmi"].astype(float) < 18.5
    ).astype(int)

    encoded["remainder__is_obese"] = (
        df["bmi"].astype(float) >= 30
    ).astype(int)

    encoded["remainder__obese_smoker"] = (
        (df["bmi"].astype(float) >= 30).astype(int) * smoker_flag
    )

    encoded["remainder__children_smoker"] = (
        df["children"].astype(int) * smoker_flag
    )

    return encoded[EXPECTED_COLUMNS]


def predict(encoded_df: pd.DataFrame, model_choice: str) -> np.ndarray:
    model = load_model(MODEL_PATHS[model_choice])

    if model_choice in SCALER_PATHS:
        scaler = load_scaler(SCALER_PATHS[model_choice])
        X = scaler.transform(encoded_df)
    else:
        X = encoded_df

    return model.predict(X)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return output.getvalue()


# ----------------------------
# Explanation logic
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
            f" \${gap:,.0f} more per year than non-smokers: "
            f" \${stats['smoker_avg']:,.0f}  vs $\{stats['nonsmoker_avg']:,.0f} "
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
    Returns a dataframe of exact dollar contribution per feature.
    """
    coefs = model.coef_
    intercept = model.intercept_
    values = encoded_row.iloc[0]

    contributions = []
    for col, coef in zip(EXPECTED_COLUMNS, coefs):
        contribution = coef * values[col]
        contributions.append({
            "Feature": FEATURE_LABELS.get(col, col),
            "Contribution ($)": contribution
        })

    contrib_df = pd.DataFrame(contributions)
    contrib_df.loc[len(contrib_df)] = {
        "Feature": "Base amount (intercept)",
        "Contribution ($)": intercept
    }
    contrib_df["Abs"] = contrib_df["Contribution ($)"].abs()
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


def render_explanation(raw_input, encoded_row, prediction, model_choice, model, stats):
    st.markdown("### 🧠 Why this estimate?")

    # Rule-based comparison — always shown, works for every model
    points = build_rule_based_explanation(
        age=raw_input["age"],
        sex=raw_input["sex"],
        bmi=raw_input["bmi"],
        children=raw_input["children"],
        smoker=raw_input["smoker"],
        region=raw_input["region"],
        prediction=prediction,
        stats=stats,
    )
    for p in points:
        st.markdown(f"- {p}")

    # Model-specific deeper insight
    with st.expander("🔬 See model-specific details"):
        if model_choice == "Multiple Linear Regression":
            st.write(
                "Linear Regression assigns a fixed dollar weight to each "
                "factor. Here's the exact breakdown for your inputs:"
            )
            try:
                contrib_df = linear_regression_breakdown(encoded_row, model)
                st.dataframe(
                    contrib_df.style.format({"Contribution ($)": "${:,.2f}"}),
                    use_container_width=True
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


# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧮 Calculator"])

# ----------------------------
# HOME PAGE
# ----------------------------
if page == "🏠 Home":
    st.title("🏥 Medical Insurance Cost Predictor")

    st.markdown(
        """
        Welcome! This tool estimates **medical insurance charges** based on
        personal and lifestyle factors, using machine learning models trained
        on real insurance data.

        ### 🔍 What this app does
        - Predicts your estimated **annual medical insurance cost**
        - Explains **why your cost is high or low** compared to typical values
        - Lets you compare predictions from **three different ML models**:
            - **Random Forest**
            - **Multiple Linear Regression**
            - **SVR (RBF Kernel)**
        - Supports **single manual entry** and **bulk Excel upload**

        ### 🧭 How to use the Calculator
        1. Click **"🧮 Calculator"** in the sidebar.
        2. Select a **model**.
        3. Choose **Manual Entry** or **Excel Upload**.
        4. After predicting, check the **"Why this estimate?"** section to
            understand which factors pushed your cost up or down.

        ### ⚠️ Disclaimer
        This tool is for **educational and illustrative purposes only** and
        is not a substitute for an actual insurance quote.
        """
    )

    st.info("👈 Use the sidebar to get started with the Calculator.")

# ----------------------------
# CALCULATOR PAGE
# ----------------------------
elif page == "🧮 Calculator":
    st.title("🧮 Insurance Cost Calculator")

    stats = load_benchmark_stats()

    model_choice = st.selectbox("Select Prediction Model", list(MODEL_PATHS.keys()))

    st.divider()

    input_mode = st.radio(
        "Input Mode", ["✍️ Manual Entry"], horizontal=True
    )

    st.divider()

    # ------------------------
    # MANUAL ENTRY MODE
    # ------------------------
    if input_mode == "✍️ Manual Entry":
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 18, 100, 30)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            children = st.number_input("Children", 0, 10, 0)

        with col2:
            sex = st.selectbox("Sex", ["Male", "Female"])
            smoker = st.selectbox("Smoker", ["Yes", "No"])
            region = st.selectbox(
                "Region", ["Northeast", "Northwest", "Southeast", "Southwest"]
            )

        if st.button("Predict", type="primary"):
            try:
                raw_input = {
                    "age": age, "sex": sex, "bmi": bmi,
                    "children": children, "smoker": smoker, "region": region,
                }
                raw_df = pd.DataFrame([raw_input])
                encoded_df = preprocess_raw_df(raw_df)
                prediction = predict(encoded_df, model_choice)[0]

                st.success(
                    f"**{model_choice}** estimated insurance cost: "
                    f"**${prediction:,.2f}**"
                )

                model = load_model(MODEL_PATHS[model_choice])
                render_explanation(
                    raw_input, encoded_df, prediction, model_choice, model, stats
                )

            except FileNotFoundError as e:
                st.error(f"Model file not found for **{model_choice}**: {e}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    