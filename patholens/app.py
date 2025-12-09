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
from models.pcam_model import PCamPredictor, load_pcam_model
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
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    * {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1rem;
        color: #718096;
        margin-bottom: 2rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #2d3748;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
    }
    .info-box {
        background-color: #f7fafc;
        border-left: 3px solid #4a5568;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 6px 6px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .warning-box {
        background-color: #f7fafc;
        border-left: 3px solid #a0aec0;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 6px 6px 0;
        color: #4a5568;
    }
    .warning-box strong {
        color: #4a5568;
        font-weight: 600;
    }
    .grade-bar {
        height: 20px;
        border-radius: 3px;
        margin: 4px 0;
    }
    .stButton>button {
        background-color: #4a5568;
        color: #f7fafc;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 0.9rem;
        font-weight: 500;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #718096;
    }
    .disclaimer {
        background-color: #f7fafc;
        border: 1px solid #a0aec0;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.8rem;
        color: #4a5568;
    }
    .disclaimer strong {
        color: #4a5568;
        font-weight: 600;
    }

    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Override Streamlit defaults */
    .stMarkdown, .stText, p, span, label, .stSelectbox, .stSlider {
        font-family: 'JetBrains Mono', monospace !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'JetBrains Mono', monospace !important;
        color: #2d3748;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #2d3748;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #718096;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(use_pcam: bool = True):
    """Load and cache the ML models."""
    device = get_device()
    if use_pcam:
        predictor = load_pcam_model(device=device)
        return predictor, device, 'pcam'
    else:
        classifier_model = load_cancer_classifier(device=device)
        return classifier_model, device, 'resnet'


def create_sidebar():
    """Create the sidebar with settings and information."""
    with st.sidebar:
        st.markdown("### Model Selection")

        use_pcam = st.checkbox(
            "Use PCam Model",
            value=True,
            help="Use PatchCamelyon-optimised model (96x96 patches)"
        )

        st.markdown("---")
        st.markdown("### Analysis Settings")

        if use_pcam:
            patch_size = st.slider(
                "Patch Size",
                min_value=64,
                max_value=128,
                value=96,
                step=16,
                help="PCam uses 96x96 patches"
            )
            stride = st.slider(
                "Stride",
                min_value=24,
                max_value=96,
                value=48,
                step=12,
                help="Smaller stride = denser, smoother heatmap"
            )
        else:
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
                help="Step size between patches"
            )

        min_tissue = st.slider(
            "Minimum Tissue Fraction",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Minimum tissue content required in patch"
        )

        heatmap_alpha = st.slider(
            "Heatmap Opacity",
            min_value=0.2,
            max_value=0.8,
            value=0.6,
            step=0.1,
            help="Transparency of heatmap overlay"
        )

        st.markdown("---")
        st.markdown("### Model Information")

        if use_pcam:
            st.markdown("**Model:** PCam ResNet18")
            st.markdown("**Input:** 96x96 patches")
            st.markdown("**Task:** Tumour detection")
            st.markdown("**Segmentation:** Colour deconvolution")
        else:
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
            'heatmap_alpha': heatmap_alpha,
            'use_pcam': use_pcam
        }


def run_analysis(
    image: Image.Image,
    model_or_predictor,
    device: torch.device,
    settings: Dict
) -> Dict:
    """
    Run the complete analysis pipeline.

    Args:
        image: Input histopathology image
        model_or_predictor: Loaded classifier model or PCamPredictor
        device: Torch device
        settings: Analysis settings from sidebar

    Returns:
        Dictionary containing all analysis results
    """
    results = {}
    use_pcam = settings.get('use_pcam', True)

    # Initialise components based on model type
    # Check if it's already a predictor with predict_batch method
    if use_pcam and hasattr(model_or_predictor, 'predict_batch') and hasattr(model_or_predictor, 'model'):
        # It's a PCamPredictor - use directly
        predictor = model_or_predictor
    elif hasattr(model_or_predictor, 'predict_batch'):
        # It's already a classifier with predict_batch
        predictor = model_or_predictor
    else:
        # It's a raw model - wrap with CancerClassifier
        predictor = CancerClassifier(model_or_predictor, device)

    patch_extractor = PatchExtractor(
        patch_size=settings['patch_size'],
        stride=settings['stride'],
        min_tissue_fraction=settings['min_tissue']
    )
    segmenter = NucleiSegmenter()
    heatmap_generator = HeatmapGenerator(
        alpha=settings['heatmap_alpha'],
        smooth_sigma=12.0,  # Smoother heatmaps
        colormap='magma'
    )
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

    batch_size = 32 if use_pcam else 16
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        batch_images = [p.image for p in batch]
        batch_results = predictor.predict_batch(batch_images)

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
        st.image(original_image, width="stretch")

    with col2:
        st.markdown("**Cancer Probability Heatmap**")
        st.image(results['heatmap_overlay'], width="stretch")

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

    # Enhanced Analysis Interpretation UI
    st.markdown("### Analysis Interpretation")

    # Extract key metrics for display
    mean_prob = results['heatmap_stats']['mean_probability']
    grade = results['grade_estimate']
    gf = results['grading_features']
    nstats = results['nuclei_stats']

    # Risk level determination
    if mean_prob < 0.3:
        risk_level = "Low"
        risk_colour = "#48bb78"
    elif mean_prob < 0.6:
        risk_level = "Moderate"
        risk_colour = "#ecc94b"
    else:
        risk_level = "Elevated"
        risk_colour = "#f56565"

    # Key Findings Cards
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; text-align: center;">
            <div style="font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Risk Assessment</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: {risk_colour};">{risk_level}</div>
            <div style="font-size: 0.8rem; color: #a0aec0;">{mean_prob:.1%} probability</div>
        </div>
        <div style="background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; text-align: center;">
            <div style="font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Predicted Grade</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #2d3748;">Grade {grade.primary_grade}</div>
            <div style="font-size: 0.8rem; color: #a0aec0;">{grade.confidence:.0%} confidence</div>
        </div>
        <div style="background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; text-align: center;">
            <div style="font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Cellular Analysis</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #2d3748;">{results['malignant_cell_count']}</div>
            <div style="font-size: 0.8rem; color: #a0aec0;">suspected malignant cells</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Analysis Flow Diagram (SVG)
    st.markdown("""
    <div style="background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0;">
        <div style="font-size: 0.85rem; font-weight: 600; color: #4a5568; margin-bottom: 1rem;">Analysis Pipeline</div>
        <svg viewBox="0 0 800 80" style="width: 100%; height: auto;">
            <!-- Boxes -->
            <rect x="0" y="20" width="120" height="40" rx="6" fill="#e2e8f0" stroke="#a0aec0"/>
            <rect x="170" y="20" width="120" height="40" rx="6" fill="#e2e8f0" stroke="#a0aec0"/>
            <rect x="340" y="20" width="120" height="40" rx="6" fill="#e2e8f0" stroke="#a0aec0"/>
            <rect x="510" y="20" width="120" height="40" rx="6" fill="#e2e8f0" stroke="#a0aec0"/>
            <rect x="680" y="20" width="120" height="40" rx="6" fill="#4a5568" stroke="#2d3748"/>

            <!-- Arrows -->
            <path d="M125 40 L165 40" stroke="#a0aec0" stroke-width="2" marker-end="url(#arrow)"/>
            <path d="M295 40 L335 40" stroke="#a0aec0" stroke-width="2" marker-end="url(#arrow)"/>
            <path d="M465 40 L505 40" stroke="#a0aec0" stroke-width="2" marker-end="url(#arrow)"/>
            <path d="M635 40 L675 40" stroke="#a0aec0" stroke-width="2" marker-end="url(#arrow)"/>

            <!-- Arrow marker -->
            <defs>
                <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L9,3 z" fill="#a0aec0"/>
                </marker>
            </defs>

            <!-- Labels -->
            <text x="60" y="45" text-anchor="middle" fill="#4a5568" font-size="11" font-family="JetBrains Mono, monospace">Patch Extract</text>
            <text x="230" y="45" text-anchor="middle" fill="#4a5568" font-size="11" font-family="JetBrains Mono, monospace">Classification</text>
            <text x="400" y="45" text-anchor="middle" fill="#4a5568" font-size="11" font-family="JetBrains Mono, monospace">Segmentation</text>
            <text x="570" y="45" text-anchor="middle" fill="#4a5568" font-size="11" font-family="JetBrains Mono, monospace">Grading</text>
            <text x="740" y="45" text-anchor="middle" fill="#f7fafc" font-size="11" font-family="JetBrains Mono, monospace">Result</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)

    # Detailed Findings in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; height: 100%;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #4a5568; margin-bottom: 1rem;">Nuclear Morphology</div>
        """, unsafe_allow_html=True)

        # Pleomorphism indicator
        pleo_pct = gf.nuclear_pleomorphism_score * 100
        st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                    <span>Pleomorphism</span>
                    <span>{gf.nuclear_pleomorphism_score:.2f}</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: #4a5568; height: 100%; width: {pleo_pct}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Atypia indicator
        atypia_pct = gf.nuclear_atypia_score * 100
        st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                    <span>Atypia</span>
                    <span>{gf.nuclear_atypia_score:.2f}</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: #4a5568; height: 100%; width: {atypia_pct}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Heterogeneity indicator
        het_pct = gf.texture_heterogeneity * 100
        st.markdown(f"""
            <div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                    <span>Heterogeneity</span>
                    <span>{gf.texture_heterogeneity:.2f}</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: #4a5568; height: 100%; width: {het_pct}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; height: 100%;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #4a5568; margin-bottom: 1rem;">Grade Distribution</div>
        """, unsafe_allow_html=True)

        # Grade 1
        st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                    <span>Grade 1 (Well-diff.)</span>
                    <span>{grade.grade_1:.1%}</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: #48bb78; height: 100%; width: {grade.grade_1 * 100}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Grade 2
        st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                    <span>Grade 2 (Mod.-diff.)</span>
                    <span>{grade.grade_2:.1%}</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: #ecc94b; height: 100%; width: {grade.grade_2 * 100}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Grade 3
        st.markdown(f"""
            <div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                    <span>Grade 3 (Poorly-diff.)</span>
                    <span>{grade.grade_3:.1%}</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: #f56565; height: 100%; width: {grade.grade_3 * 100}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # AI Interpretation (collapsible)
    with st.expander("View AI Interpretation", expanded=False):
        st.markdown("""
        <div style="background: #f7fafc; border-radius: 8px; padding: 1rem; font-size: 0.85rem; line-height: 1.6; color: #4a5568;">
        """, unsafe_allow_html=True)
        st.write(results['explanation'])
        st.markdown("</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="background: #f7fafc; border: 1px solid #a0aec0; border-radius: 6px; padding: 1rem; margin-top: 1.5rem; font-size: 0.8rem; color: #4a5568;">
        <strong style="color: #4a5568;">Research Use Only</strong> &mdash; This analysis is generated by a computational model
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

    # Sidebar settings (must come first to determine model type)
    settings = create_sidebar()

    # Load models based on settings
    with st.spinner("Loading models..."):
        model_or_predictor, device, model_type = load_models(use_pcam=settings['use_pcam'])

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
        st.image(image, caption="Uploaded Image", width="stretch")
    elif use_sample and sample_path.exists():
        image = Image.open(sample_path).convert('RGB')
        st.image(image, caption="Sample Image", width="stretch")
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
                width="stretch"
            )

        if analyse_button:
            with st.spinner("Running analysis pipeline..."):
                results = run_analysis(image, model_or_predictor, device, settings)

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
