import streamlit as st
import torch
import json
import numpy as np
from model import MLP

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Monkeypox Prediction (PFL+ES)",
    page_icon="🩺",
    layout="centered"
)

DEVICE = torch.device("cpu")
DROPOUT_RATE = 0.3

STATE_MODELS = {
    "Abia": "saved_models/abia_best_model.pt",
    "Anambra": "saved_models/anambra_best_model.pt",
    "Enugu": "saved_models/enugu_best_model.pt",
    "Imo": "saved_models/imo_best_model.pt",
    "Ebonyi": "saved_models/ebonyi_best_model.pt"
}

# -----------------------------
# LOAD FEATURE NAMES
# -----------------------------
@st.cache_data
def load_feature_names():
    """Load feature names from config JSON"""
    with open("config/feature_names.json", "r") as f:
        data = json.load(f)
    return data["features"]

FEATURE_NAMES = load_feature_names()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model(state_name):
    """Load the trained model for a specific state"""
    model = MLP(
        input_dim=len(FEATURE_NAMES),
        dropout_rate=DROPOUT_RATE
    )
    model.load_state_dict(
        torch.load(STATE_MODELS[state_name], map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model

# -----------------------------
# UI HEADER
# -----------------------------
st.title("🩺 Monkeypox Prediction System")
st.caption("Personalized Federated Learning with Evolutionary Strategies")

st.markdown("""
This application uses state-specific AI models trained collaboratively across five states 
in southeastern Nigeria to predict monkeypox test outcomes based on patient symptoms.
""")

st.markdown("---")

# -----------------------------
# STATE SELECTION
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    state = st.selectbox(
        "🏥 Select State Model",
        list(STATE_MODELS.keys()),
        help="Choose the state-specific model for prediction"
    )

with col2:
    st.metric("Selected State", state)

# Load the selected model
model = load_model(state)

st.markdown("---")

# -----------------------------
# PATIENT SYMPTOM INPUTS
# -----------------------------
st.subheader("📋 Patient Information")
st.write("Please indicate the presence of each symptom:")

inputs = {}

# Filter out symptom_count - we'll calculate it automatically
symptom_features = [f for f in FEATURE_NAMES if f != "symptom_count"]

# Create a more organized layout with multiple columns
num_cols = 2
features_per_col = len(symptom_features) // num_cols + (len(symptom_features) % num_cols > 0)

cols = st.columns(num_cols)

for idx, feature in enumerate(symptom_features):
    col_idx = idx // features_per_col
    with cols[col_idx]:
        # Clean up feature name for display
        display_name = feature.replace("_", " ").title()
        inputs[feature] = st.selectbox(
            display_name,
            options=["No", "Yes"],
            key=f"input_{feature}"
        )

st.markdown("---")

# -----------------------------
# INFERENCE
# -----------------------------
if st.button("🔬 Run Prediction", type="primary", use_container_width=True):
    # Calculate symptom_count automatically
    symptom_count = sum(1 for f in symptom_features if inputs[f] == "Yes")
    
    # Convert inputs to binary array in the correct order
    X = []
    for feature in FEATURE_NAMES:
        if feature == "symptom_count":
            X.append(symptom_count)
        else:
            X.append(1 if inputs[feature] == "Yes" else 0)
    
    X = np.array([X])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        logit = model(X_tensor)
        prob = torch.sigmoid(logit).item()
    
    prediction = "Positive" if prob >= 0.5 else "Negative"
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    
    # Create columns for results
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == "Positive":
            st.error(f"**Predicted Outcome:** {prediction}")
        else:
            st.success(f"**Predicted Outcome:** {prediction}")
    
    with res_col2:
        st.metric("Risk Probability", f"{prob:.2%}")
    
    # Add interpretation
    st.markdown("---")
    st.subheader("🔍 Interpretation")
    
    if prediction == "Positive":
        st.warning(
            "⚠️ **High Risk**: The model indicates a higher likelihood of monkeypox. "
            "Please consult with a healthcare provider for laboratory confirmation."
        )
    else:
        st.success(
            "✅ **Low Risk**: The model indicates a lower likelihood of monkeypox. "
            "However, if symptoms persist, please seek medical attention."
        )
    
    # Confidence indicator
    confidence = abs(prob - 0.5) * 2  # Scale to 0-1 where 0.5 prob = 0 confidence
    st.progress(confidence)
    st.caption(f"Prediction confidence: {confidence:.1%}")
    
    # Disclaimer
    st.info(
        "⚕️ **Disclaimer**: This tool is a Ph.D. artefact designed by Anozie Onyezewe for decision support only and should not replace "
        "professional medical diagnosis. Always consult qualified healthcare providers."
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Powered by Federated Learning | Model trained on data from Abia, Anambra, Enugu, Imo, and Ebonyi states</p>
    </div>
    """,
    unsafe_allow_html=True
)