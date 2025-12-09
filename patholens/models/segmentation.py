import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage import morphology, measure

# --- Try importing StarDist (The "Gold Standard" method) ---
try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    HAS_STARDIST = True
except ImportError:
    HAS_STARDIST = False
    print("⚠️ StarDist not found. Using fallback watershed segmentation.")

class NucleiSegmenter:
    def __init__(self):
        self.stardist_model = None
        
        if HAS_STARDIST:
            try:
                # Automatically downloads the '2D_versatile_he' model on first run
                # This model is specifically trained on H&E images
                print("🌟 Loading StarDist H&E model...")
                self.stardist_model = StarDist2D.from_pretrained('2D_versatile_he')
            except Exception as e:
                print(f"⚠️ Failed to load StarDist: {e}")
                self.stardist_model = None

    def segment(self, image):
        """
        Main segmentation router. 
        Attempts StarDist first (High Accuracy), falls back to Watershed.
        """
        if self.stardist_model is not None:
            return self._segment_stardist(image)
        else:
            return self._segment_watershed(image)

    def _segment_stardist(self, image):
        """AI-based segmentation using StarDist."""
        img_np = np.array(image)
        
        # StarDist requires normalization
        # We normalize channels independently (axis=0,1 for H,W,C is tricky, 
        # usually stardist handles (H,W,C))
        img_norm = normalize(img_np, 1, 99.8, axis=(0,1))
        
        # Predict
        # labels is the mask where each nucleus has a unique ID
        labels, details = self.stardist_model.predict_instances(img_norm)
        
        # Generate individual binary masks (optional, usually labels are enough)
        # For compatibility with existing pipeline, we return the label map
        # If your pipeline expects list of masks:
        nuclei_masks = []
        # Optimisation: Don't extract every single mask if we just need stats.
        # But if we need visualization:
        unique_ids = np.unique(labels)
        if len(unique_ids) > 1: # 0 is background
             # Just return labels to save memory, or implement if visualization needs specific format
             pass
             
        # Compatible return: (masks_list, labels_map, details)
        return [], labels, details

    def _segment_watershed(self, image):
        """Classic Computer Vision fallback (Robust but less accurate)."""
        img_np = np.array(image)
        
        # 1. Color Deconvolution approx (Blue - Red)
        red = img_np[:,:,0].astype(float)
        blue = img_np[:,:,2].astype(float)
        nuclei_map = blue - red
        
        # 2. Thresholding
        nuclei_map = np.clip(nuclei_map, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(nuclei_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Cleanup
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=20)
        binary = morphology.binary_opening(binary, morphology.disk(2))
        
        # 4. Watershed
        distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
        # Handle cases with no nuclei
        if distance.max() == 0:
            return [], np.zeros(img_np.shape[:2], dtype=int), None
            
        _, local_max = cv2.threshold(distance, 0.4 * distance.max(), 255, 0)
        
        num_labels, labels = cv2.connectedComponents(local_max.astype(np.uint8))
        markers = cv2.watershed(img_np, labels.astype(np.int32))
        
        # Clean labels
        clean_labels = np.zeros_like(labels)
        clean_labels[markers > 1] = markers[markers > 1] # Skip background and borders
        
        return [], clean_labels, None

def compute_nuclei_statistics(nuclei_input):
    """
    Computes grading statistics. 
    Accepts either a list of masks OR a label map (from StarDist).
    """
    # Check if input is a label map (numpy array) or list of masks
    if isinstance(nuclei_input, np.ndarray):
        # Input is a label map (H, W) where pixels are IDs
        label_map = nuclei_input
        if label_map.max() == 0:
            return {'count': 0, 'mean_area': 0, 'std_area': 0}
            
        # Extract properties using skimage.measure
        props = measure.regionprops(label_map)
        areas = [p.area for p in props]
        
        return {
            'count': len(areas),
            'mean_area': np.mean(areas),
            'std_area': np.std(areas)
        }
    
    else:
        # Input is list of masks (Fallback/Legacy)
        nuclei_list = nuclei_input
        if not nuclei_list:
            return {'count': 0, 'mean_area': 0, 'std_area': 0}
            
        areas = [np.sum(n > 0) for n in nuclei_list]
        return {
            'count': len(nuclei_list),
            'mean_area': np.mean(areas),
            'std_area': np.std(areas)
        }

# Alias
def segment_nuclei(image):
    seg = NucleiSegmenter()
    return seg.segment(image)