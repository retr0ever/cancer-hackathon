# config/styles.py
import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        * { font-family: 'JetBrains Mono', monospace !important; }
        
        .main-header {
            font-size: 2.2rem; font-weight: 600; color: #2d3748;
            margin-bottom: 0.5rem; letter-spacing: -0.02em;
        }
        .sub-header {
            font-size: 1rem; color: #718096; margin-bottom: 2rem;
        }
        /* ... [Include the rest of your CSS here] ... */
        
        .grade-bar { height: 20px; border-radius: 3px; margin: 4px 0; }
    </style>
    """, unsafe_allow_html=True)