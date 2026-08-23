import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the trained model with caching for better performance"""
    model_path = Path("models/random_forest_model.pkl")
    if not model_path.exists():
        st.error("Model file not found. Please ensure 'models/random_forest_model.pkl' exists.")
        return None
    return joblib.load(model_path)

# Cache the feature names
@st.cache_data
def get_feature_columns():
    """Define the expected feature columns for the model"""
    return [
        "cat__sex_male",
        "cat__smoker_yes",
        "cat__region_northwest",
        "cat__region_southeast",
        "cat__region_southwest",
        "remainder__age",
        "remainder__bmi",
        "remainder__children"
    ]

# Load model
model = load_model()
feature_columns = get_feature_columns()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #27ae60;
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Medical Insurance Cost Prediction</h1>', unsafe_allow_html=True)

# Description
with st.expander("ℹ️ About this app", expanded=False):
    st.markdown("""
    This app predicts medical insurance costs based on personal information.
    The prediction is made using a Random Forest model trained on historical insurance data.
    
    **Note:** This is a demonstration tool and should not be used for actual insurance decisions.
    """)

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    
    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        help="Your age in years"
    )
    
    sex = st.radio(
        "Sex",
        ["Male", "Female"],
        horizontal=True
    )
    
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        format="%.1f",
        help="BMI = weight(kg) / height(m)²"
    )

with col2:
    st.subheader("Lifestyle & Location")
    
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )
    
    smoker = st.toggle(
        "Smoker",
        value=False,
        help="Toggle if you are a smoker"
    )
    
    region = st.selectbox(
        "Region",
        [
            "Northeast",
            "Northwest",
            "Southeast",
            "Southwest"
        ],
        help="US region where you live"
    )

# Add BMI category indicator
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "⚠️"
    elif bmi < 25:
        return "Normal weight", "✅"
    elif bmi < 30:
        return "Overweight", "⚠️"
    else:
        return "Obese", "❌"

bmi_category, bmi_emoji = get_bmi_category(bmi)
st.info(f"BMI Category: {bmi_emoji} {bmi_category}")

# Prediction section
st.markdown("---")

# Add a predict button with more style
col_button, col_spacer, col_result = st.columns([1, 1, 2])

with col_button:
    predict_button = st.button(
        "💰 Predict Insurance Cost",
        type="primary",
        use_container_width=True
    )

if predict_button:
    if model is None:
        st.error("Model not loaded. Please check the model file.")
    else:
        # Prepare input data
        input_data = pd.DataFrame({
            "cat__sex_male": [1 if sex == "Male" else 0],
            "cat__smoker_yes": [1 if smoker else 0],
            "cat__region_northwest": [1 if region == "Northwest" else 0],
            "cat__region_southeast": [1 if region == "Southeast" else 0],
            "cat__region_southwest": [1 if region == "Southwest" else 0],
            "remainder__age": [age],
            "remainder__bmi": [bmi],
            "remainder__children": [children]
        })
        
        # Ensure correct column order
        input_data = input_data[feature_columns]
        
        # Make prediction
        try:
            prediction = model.predict(input_data)
            prediction_value = prediction[0]
            
            # Display result with style
            st.markdown("---")
            st.markdown("### 📊 Prediction Result")
            
            # Show the prediction in a nice box
            st.markdown(f"""
                <div class="prediction-box">
                    Estimated Annual Insurance Cost<br>
                    <strong>${prediction_value:,.2f}</strong>
                </div>
            """, unsafe_allow_html=True)
            
            # Show input summary
            with st.expander("📋 View Input Summary", expanded=False):
                summary_data = {
                    "Feature": ["Age", "Sex", "BMI", "Children", "Smoker", "Region"],
                    "Value": [age, sex, f"{bmi:.1f} ({bmi_category})", children, "Yes" if smoker else "No", region]
                }
                summary_df = pd.DataFrame(summary_data)
                st.table(summary_df)
            
            # Add a disclaimer
            st.markdown("""
                <div class="info-box">
                    <strong>⚠️ Disclaimer:</strong> This is a predictive model and should be used for 
                    informational purposes only. Actual insurance costs may vary based on many factors 
                    not captured by this model.
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# Add footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Data source: Medical Insurance Dataset")