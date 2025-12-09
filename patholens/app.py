import ssl
import certifi
import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Fix SSL certificates (common macOS issue)
try:
    ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
except Exception:
    ssl._create_default_https_context = ssl._create_unverified_context

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import Modules
from config.styles import apply_custom_styles
from ui.layout import setup_page, render_header, render_disclaimer
from ui.sidebar import render_sidebar
from ui.results import render_results_dashboard
from core.state import load_app_models
from core.pipeline import run_analysis_pipeline

def main():
    setup_page()
    apply_custom_styles()
    render_header()
    render_disclaimer()

    # Sidebar returns settings dict
    settings = render_sidebar()

    # Model Loading
    with st.spinner("Loading models..."):
        model, device = load_app_models(use_pcam=settings['use_pcam'])

    # --- Image Selection Section ---
    st.markdown("### Upload Histopathology Image")

    # 1. File Upload
    uploaded_file = st.file_uploader(
        "Upload an H&E stained tissue section (PNG/JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a histopathology image for analysis"
    )

    # 2. Sample Image Checkbox
    sample_path = Path(__file__).parent / "assets" / "sample_histo.png"
    use_sample = st.checkbox(
        "Use sample image",
        value=False,
        help="Use the included sample histopathology image"
    )

    image = None

    # 3. Logic to determine which image to use
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        # FIXED: Removed width=None, added use_container_width=True
        st.image(image, caption="Uploaded Image", use_container_width=True)
    elif use_sample:
        if sample_path.exists():
            image = Image.open(sample_path).convert('RGB')
            # FIXED: Removed width=None, added use_container_width=True
            st.image(image, caption="Sample Image", use_container_width=True)
        else:
            st.warning(
                f"Sample image not found at {sample_path}. "
                "Please add a sample image or upload your own."
            )

    # --- Analysis Section ---
    if image is not None:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            analyse_btn = st.button("Analyse Slide", type="primary", use_container_width=True)
        
        if analyse_btn:
            # Progress tracking UI
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_ui_progress(percent, message):
                progress_bar.progress(percent)
                status_text.text(message)

            with st.spinner("Running analysis pipeline..."):
                results = run_analysis_pipeline(image, model, settings, update_ui_progress)
                
                if results:
                    # Save to session state
                    st.session_state['results'] = results
                    st.session_state['original_image'] = image
                    st.rerun()

    # --- Results Persistence ---
    if 'results' in st.session_state and 'original_image' in st.session_state:
        render_results_dashboard(
            st.session_state['results'], 
            st.session_state['original_image']
        )

if __name__ == "__main__":
    main()