import streamlit as st
from models.load_model import load_cancer_classifier, get_device
from models.classifier import CancerClassifier
# corrected import location below:
from models.pcam_model import load_pcam_model

@st.cache_resource
def load_app_models(use_pcam: bool = True):
    """Load and cache the ML models."""
    device = get_device()
    
    if use_pcam:
        # Now importing correctly from pcam_model
        predictor = load_pcam_model(device=device)
        return predictor, device
    else:
        classifier_model = load_cancer_classifier(device=device)
        # Wrap immediately for consistency
        predictor = CancerClassifier(classifier_model, device)
        return predictor, device