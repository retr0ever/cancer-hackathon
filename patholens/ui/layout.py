# ui/layout.py
import streamlit as st

def setup_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="PathoLens",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_header():
    """Render the main application header."""
    st.markdown('<h1 class="main-header">PathoLens</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Computational Histopathology Analysis for Cancer Detection</p>',
        unsafe_allow_html=True
    )

def render_disclaimer():
    """Render the top warning/disclaimer box."""
    st.markdown("""
    <div class="warning-box">
        <strong>Research Demonstration Tool</strong><br>
        This tool is designed for research purposes only and is not intended for clinical use.
        All outputs are for demonstration and should not be used for medical decisions.
    </div>
    """, unsafe_allow_html=True)