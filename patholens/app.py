"""
PathoLens - Computational Histopathology Analysis Tool

A research demonstration tool for analysing H&E stained histopathology
images to identify cancerous regions, estimate malignant cell density,
and provide explainable insights.

DISCLAIMER: This tool is for research purposes only and is not intended
for clinical diagnosis or treatment decisions.

Run with: streamlit run app.py
"""

# Fix SSL certificates before any other imports (common macOS issue)
import ssl
import certifi
try:
    ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
except Exception:
    ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import numpy as np
from PIL import Image
import torch
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.load_model import load_cancer_classifier, get_device, get_model_info
from models.classifier import CancerClassifier
from models.segmentation import NucleiSegmenter, compute_nuclei_statistics
from models.heatmap import HeatmapGenerator, PatchPrediction
from utils.patching import PatchExtractor
from utils.grading import TumourGrader, format_grade_report
from utils.explanation import generate_explanation, format_features_for_display


# Page configuration
st.set_page_config(
    page_title="PathoLens",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a4a6a;
        margin-bottom: 2rem;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .grade-bar {
        height: 24px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .disclaimer {
        background-color: #ffe0e0;
        border: 1px solid #ff6b6b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load and cache the ML models."""
    device = get_device()
    classifier_model = load_cancer_classifier(device=device)
    return classifier_model, device


def create_sidebar():
    """Create the sidebar with settings and information."""
    with st.sidebar:
        st.markdown("### Analysis Settings")

        patch_size = st.slider(
            "Patch Size",
            min_value=128,
            max_value=512,
            value=224,
            step=32,
            help="Size of patches for analysis (pixels)"
        )

        stride = st.slider(
            "Stride",
            min_value=32,
            max_value=224,
            value=112,
            step=16,
            help="Step size between patches (smaller = more overlap)"
        )

        min_tissue = st.slider(
            "Minimum Tissue Fraction",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum tissue content required in patch"
        )

        heatmap_alpha = st.slider(
            "Heatmap Opacity",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Transparency of heatmap overlay"
        )

        st.markdown("---")
        st.markdown("### Model Information")

        model_info = get_model_info()
        st.markdown(f"**Classifier:** {model_info['classifier']['architecture']}")
        st.markdown(f"**Input Size:** {model_info['classifier']['input_size'][0]}x{model_info['classifier']['input_size'][1]}")
        st.markdown(f"**Segmentation:** {model_info['segmentation']['method'][:30]}...")

        st.markdown("---")
        st.markdown("### About PathoLens")
        st.markdown("""
        PathoLens is a computational histopathology
        research tool that analyses H&E stained
        tissue sections to identify potential
        cancerous regions.

        **For research use only.**
        """)

        return {
            'patch_size': patch_size,
            'stride': stride,
            'min_tissue': min_tissue,
            'heatmap_alpha': heatmap_alpha
        }


def run_analysis(
    image: Image.Image,
    classifier_model: torch.nn.Module,
    device: torch.device,
    settings: Dict
) -> Dict:
    """
    Run the complete analysis pipeline.

    Args:
        image: Input histopathology image
        classifier_model: Loaded classifier model
        device: Torch device
        settings: Analysis settings from sidebar

    Returns:
        Dictionary containing all analysis results
    """
    results = {}

    # Initialise components
    classifier = CancerClassifier(classifier_model, device)
    patch_extractor = PatchExtractor(
        patch_size=settings['patch_size'],
        stride=settings['stride'],
        min_tissue_fraction=settings['min_tissue']
    )
    segmenter = NucleiSegmenter()
    heatmap_generator = HeatmapGenerator(alpha=settings['heatmap_alpha'])
    grader = TumourGrader()

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Extract patches
    status_text.text("Extracting tissue patches...")
    patches = patch_extractor.extract_patches(image)
    results['num_patches'] = len(patches)
    progress_bar.progress(20)

    if not patches:
        st.warning("No tissue patches detected. Try adjusting the tissue threshold.")
        return None

    # Step 2: Classify patches
    status_text.text(f"Classifying {len(patches)} patches...")
    predictions = []
    patch_probs = []

    batch_size = 16
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        batch_images = [p.image for p in batch]
        batch_results = classifier.predict_batch(batch_images)

        for j, (prob, pred_class) in enumerate(batch_results):
            patch = batch[j]
            predictions.append(PatchPrediction(
                x=patch.x,
                y=patch.y,
                width=patch.width,
                height=patch.height,
                cancer_probability=prob,
                predicted_class=pred_class
            ))
            patch_probs.append(prob)

        progress = 20 + int(60 * (i + len(batch)) / len(patches))
        progress_bar.progress(min(progress, 80))

    results['predictions'] = predictions
    results['patch_probabilities'] = patch_probs
    progress_bar.progress(80)

    # Step 3: Generate heatmap
    status_text.text("Generating probability heatmap...")
    overlay, prob_map, heatmap_stats = heatmap_generator.generate(image, predictions)
    results['heatmap_overlay'] = overlay
    results['probability_map'] = prob_map
    results['heatmap_stats'] = heatmap_stats
    progress_bar.progress(85)

    # Step 4: Nuclei segmentation
    status_text.text("Segmenting nuclei...")
    nuclei, labels, _ = segmenter.segment(image)
    nuclei_stats = compute_nuclei_statistics(nuclei)
    results['nuclei'] = nuclei
    results['nuclei_labels'] = labels
    results['nuclei_stats'] = nuclei_stats
    progress_bar.progress(90)

    # Step 5: Estimate malignant cell count
    mean_prob = heatmap_stats['mean_probability']
    malignant_count = int(len(nuclei) * mean_prob)
    uncertainty = 0.15  # Base uncertainty for simplified method
    results['malignant_cell_count'] = malignant_count
    results['malignant_cell_uncertainty'] = uncertainty

    # Step 6: Grade estimation
    status_text.text("Estimating tumour grade...")
    grading_features = grader.extract_grading_features(
        heatmap_stats, nuclei_stats, patch_probs
    )
    grade_estimate = grader.estimate_grade(grading_features)
    results['grade_estimate'] = grade_estimate
    results['grading_features'] = grading_features
    progress_bar.progress(95)

    # Step 7: Generate explanation
    status_text.text("Generating explanation...")
    explanation_features = {
        'cancer_probability': heatmap_stats['mean_probability'],
        'malignant_cell_count': malignant_count,
        'malignant_cell_uncertainty': uncertainty,
        'grade_1': grade_estimate.grade_1,
        'grade_2': grade_estimate.grade_2,
        'grade_3': grade_estimate.grade_3,
        'primary_grade': grade_estimate.primary_grade,
        'high_risk_regions': int(heatmap_stats['high_risk_fraction'] * len(patches)),
        'nuclear_pleomorphism_score': grading_features.nuclear_pleomorphism_score,
        'nuclear_atypia_score': grading_features.nuclear_atypia_score,
        'texture_heterogeneity': grading_features.texture_heterogeneity,
        'hotspot_locations': []
    }
    explanation = generate_explanation(explanation_features, use_api=True)
    results['explanation'] = explanation

    progress_bar.progress(100)
    status_text.text("Analysis complete!")

    return results


def display_results(results: Dict, original_image: Image.Image):
    """Display analysis results in the UI."""

    st.markdown("---")
    st.markdown("## Analysis Results")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Cancer Probability",
            f"{results['heatmap_stats']['mean_probability']:.1%}",
            delta=None
        )

    with col2:
        st.metric(
            "Malignant Cells",
            f"{results['malignant_cell_count']}",
            delta=f"±{results['malignant_cell_uncertainty']:.0%}"
        )

    with col3:
        st.metric(
            "Patches Analysed",
            f"{results['num_patches']}"
        )

    with col4:
        st.metric(
            "Nuclei Detected",
            f"{results['nuclei_stats']['count']}"
        )

    st.markdown("---")

    # Image comparison
    st.markdown("### Heatmap Overlay")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(original_image, use_container_width=True)

    with col2:
        st.markdown("**Cancer Probability Heatmap**")
        st.image(results['heatmap_overlay'], use_container_width=True)

    # Colour scale legend
    st.markdown("""
    <div style="display: flex; align-items: center; margin: 1rem 0;">
        <span style="margin-right: 1rem;">Low Risk</span>
        <div style="flex: 1; height: 20px; background: linear-gradient(to right, #313695, #4575b4, #74add1, #abd9e9, #e0f3f8, #ffffbf, #fee090, #fdae61, #f46d43, #d73027, #a50026); border-radius: 4px;"></div>
        <span style="margin-left: 1rem;">High Risk</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Grade estimation
    st.markdown("### Tumour Grade Estimation")

    grade = results['grade_estimate']

    col1, col2 = st.columns([2, 1])

    with col1:
        # Grade probability bars
        st.markdown(f"**Grade 1** (Well-differentiated): {grade.grade_1:.1%}")
        st.progress(grade.grade_1)

        st.markdown(f"**Grade 2** (Moderately-differentiated): {grade.grade_2:.1%}")
        st.progress(grade.grade_2)

        st.markdown(f"**Grade 3** (Poorly-differentiated): {grade.grade_3:.1%}")
        st.progress(grade.grade_3)

    with col2:
        st.markdown("#### Primary Estimate")
        st.markdown(f"## Grade {grade.primary_grade}")
        st.markdown(f"Confidence: {grade.confidence:.1%}")

    st.markdown("---")

    # Detailed statistics
    st.markdown("### Detailed Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Heatmap Statistics")
        stats = results['heatmap_stats']
        st.markdown(f"- Mean probability: {stats['mean_probability']:.2%}")
        st.markdown(f"- Max probability: {stats['max_probability']:.2%}")
        st.markdown(f"- High-risk fraction: {stats['high_risk_fraction']:.2%}")
        st.markdown(f"- Moderate-risk fraction: {stats['moderate_risk_fraction']:.2%}")
        st.markdown(f"- Low-risk fraction: {stats['low_risk_fraction']:.2%}")

    with col2:
        st.markdown("#### Nuclear Features")
        nstats = results['nuclei_stats']
        gf = results['grading_features']
        st.markdown(f"- Total nuclei: {nstats['count']}")
        st.markdown(f"- Mean area: {nstats['mean_area']:.1f} px")
        st.markdown(f"- Pleomorphism score: {gf.nuclear_pleomorphism_score:.2f}")
        st.markdown(f"- Atypia score: {gf.nuclear_atypia_score:.2f}")
        st.markdown(f"- Texture heterogeneity: {gf.texture_heterogeneity:.2f}")

    st.markdown("---")

    # Explanation box
    st.markdown("### Analysis Interpretation")

    st.markdown("""
    <div class="info-box">
    """, unsafe_allow_html=True)

    st.text(results['explanation'])

    st.markdown("</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>Research Use Only:</strong> This analysis is generated by a computational model
        and is intended for research demonstration purposes only. It should not be used for
        clinical diagnosis, treatment decisions, or as a substitute for professional medical
        advice. All findings require validation by a qualified pathologist.
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""

    # Header
    st.markdown('<h1 class="main-header">PathoLens</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Computational Histopathology Analysis for Cancer Detection</p>',
        unsafe_allow_html=True
    )

    # Disclaimer at top
    st.markdown("""
    <div class="warning-box">
        <strong>Research Demonstration Tool</strong><br>
        This tool is designed for research purposes only and is not intended for clinical use.
        All outputs are for demonstration and should not be used for medical decisions.
    </div>
    """, unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models..."):
        classifier_model, device = load_models()

    # Sidebar settings
    settings = create_sidebar()

    # Main content
    st.markdown("### Upload Histopathology Image")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload an H&E stained tissue section (PNG/JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a histopathology image for analysis"
    )

    # Sample image option
    sample_path = Path(__file__).parent / "assets" / "sample_histo.png"
    use_sample = st.checkbox(
        "Use sample image",
        value=False,
        help="Use the included sample histopathology image"
    )

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
    elif use_sample and sample_path.exists():
        image = Image.open(sample_path).convert('RGB')
        st.image(image, caption="Sample Image", use_container_width=True)
    elif use_sample:
        st.warning(
            f"Sample image not found at {sample_path}. "
            "Please add a sample image or upload your own."
        )

    # Analyse button
    if image is not None:
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyse_button = st.button(
                "Analyse Slide",
                type="primary",
                use_container_width=True
            )

        if analyse_button:
            with st.spinner("Running analysis pipeline..."):
                results = run_analysis(image, classifier_model, device, settings)

                if results:
                    # Store results in session state for persistence
                    st.session_state['results'] = results
                    st.session_state['original_image'] = image

        # Display results if available
        if 'results' in st.session_state and 'original_image' in st.session_state:
            display_results(
                st.session_state['results'],
                st.session_state['original_image']
            )


if __name__ == "__main__":
    main()
