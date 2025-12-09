# core/pipeline.py
import torch
from typing import Dict, Optional, Callable
from PIL import Image
from models.classifier import CancerClassifier
from models.segmentation import NucleiSegmenter, compute_nuclei_statistics
from models.heatmap import HeatmapGenerator, PatchPrediction
from utils.patching import PatchExtractor
from utils.grading import TumourGrader
from utils.explanation import generate_explanation

def run_analysis_pipeline(
    image: Image.Image,
    predictor,
    settings: Dict,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict:
    """
    Core business logic. Decoupled from Streamlit widgets.
    """
    def update_status(percent, message):
        if progress_callback:
            progress_callback(percent, message)

    results = {}
    use_pcam = settings.get('use_pcam', True)

    # 1. Initialization
    patch_extractor = PatchExtractor(
        patch_size=settings['patch_size'],
        stride=settings['stride'],
        min_tissue_fraction=settings['min_tissue']
    )
    segmenter = NucleiSegmenter()
    heatmap_generator = HeatmapGenerator(alpha=settings['heatmap_alpha'], smooth_sigma=12.0)
    grader = TumourGrader()

    # 2. Extract Patches
    update_status(10, "Extracting tissue patches...")
    patches = patch_extractor.extract_patches(image)
    results['num_patches'] = len(patches)
    
    if not patches:
        return None

    # 3. Classification
    update_status(20, f"Classifying {len(patches)} patches...")
    predictions = []
    patch_probs = []
    
    batch_size = 32 if use_pcam else 16
    total_patches = len(patches)
    
    for i in range(0, total_patches, batch_size):
        batch = patches[i:i+batch_size]
        batch_images = [p.image for p in batch]
        batch_results = predictor.predict_batch(batch_images)

        for j, (prob, pred_class) in enumerate(batch_results):
            patch = batch[j]
            predictions.append(PatchPrediction(
                x=patch.x, y=patch.y, width=patch.width, height=patch.height,
                cancer_probability=prob, predicted_class=pred_class
            ))
            patch_probs.append(prob)
        
        # Calculate intermediate progress between 20 and 80
        current_progress = 20 + int(60 * (i + len(batch)) / total_patches)
        update_status(current_progress, f"Classifying batch {i//batch_size + 1}...")

    results['predictions'] = predictions
    results['patch_probabilities'] = patch_probs

    # 4. Heatmap
    # Step 3: Generate heatmap
    update_status(85, "Generating probability heatmap...")
    # UPDATED: Unpack 4 values instead of 3
    overlay_abs, overlay_rel, prob_map, heatmap_stats = heatmap_generator.generate(image, predictions)
    
    results['heatmap_overlay'] = overlay_abs        # Default absolute
    results['heatmap_overlay_relative'] = overlay_rel # Boosted visibility
    results['probability_map'] = prob_map
    results['heatmap_stats'] = heatmap_stats

    # 5. Segmentation & Stats
    update_status(90, "Segmenting nuclei...")
    nuclei, labels, _ = segmenter.segment(image)
    nuclei_stats = compute_nuclei_statistics(nuclei)
    results['nuclei_stats'] = nuclei_stats
    
    # Malignant count logic
    mean_prob = heatmap_stats['mean_probability']
    results['malignant_cell_count'] = int(len(nuclei) * mean_prob)
    results['malignant_cell_uncertainty'] = 0.15

    # 6. Grading
    update_status(95, "Estimating tumour grade...")
    grading_features = grader.extract_grading_features(heatmap_stats, nuclei_stats, patch_probs)
    grade_estimate = grader.estimate_grade(grading_features)
    results['grade_estimate'] = grade_estimate
    results['grading_features'] = grading_features

    # 7. Explanation
    update_status(98, "Generating explanation...")
    explanation_features = {
        'cancer_probability': heatmap_stats['mean_probability'],
        'malignant_cell_count': results['malignant_cell_count'],
        'primary_grade': grade_estimate.primary_grade,
        # ... map other features ...
    }
    # Mocking explanation for structure (use real import in prod)
    results['explanation'] = generate_explanation(explanation_features, use_api=True)

    update_status(100, "Analysis complete!")
    return results