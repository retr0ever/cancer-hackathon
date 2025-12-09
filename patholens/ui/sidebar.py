# ui/sidebar.py
import streamlit as st
from models.load_model import get_model_info

def render_sidebar():
    with st.sidebar:
        st.title("PathoLens")
        st.markdown("### Model Selection")

        use_pcam = st.checkbox(
            "Use PCam Model", value=True,
            help="Use PatchCamelyon-optimised model (96x96 patches)"
        )

        st.markdown("---")
        st.markdown("### Analysis Settings")

        if use_pcam:
            patch_size = st.slider("Patch Size", 64, 128, 96, 16)
            stride = st.slider("Stride", 24, 96, 48, 12)
        else:
            patch_size = st.slider("Patch Size", 128, 512, 224, 32)
            stride = st.slider("Stride", 32, 224, 112, 16)

        min_tissue = st.slider("Minimum Tissue Fraction", 0.1, 0.9, 0.3, 0.1)
        heatmap_alpha = st.slider("Heatmap Opacity", 0.2, 0.8, 0.6, 0.1)

        # Reset Logic
        st.markdown("---")
        if st.button("Reset Analysis", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        return {
            'patch_size': patch_size,
            'stride': stride,
            'min_tissue': min_tissue,
            'heatmap_alpha': heatmap_alpha,
            'use_pcam': use_pcam
        }