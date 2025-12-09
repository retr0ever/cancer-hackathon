import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter

class PatchPrediction:
    def __init__(self, x, y, width, height, cancer_probability, predicted_class):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.cancer_probability = cancer_probability
        self.predicted_class = predicted_class

class HeatmapGenerator:
    def __init__(self, alpha=0.6, smooth_sigma=5.0, colormap='jet'):
        self.alpha = alpha
        # Lower sigma = more precise/sharp. Higher = smoother/blurrier.
        self.smooth_sigma = smooth_sigma
        self.colormap = colormap

    def generate(self, image, predictions):
        """
        Generates a high-precision heatmap overlay.
        Returns: (overlay_image, raw_probability_map, statistics_dict)
        """
        if not predictions:
            # Return empty/safe defaults if no predictions
            img_np = np.array(image)
            return image, np.zeros((img_np.shape[0], img_np.shape[1])), {
                'mean_probability': 0.0, 'max_probability': 0.0,
                'high_risk_fraction': 0.0, 'moderate_risk_fraction': 0.0, 'low_risk_fraction': 0.0
            }

        img_np = np.array(image)
        height, width = img_np.shape[:2]

        # 1. Create High-Res Probability Map
        # 1/8th resolution preserves precision better than small grids
        scale_factor = 0.125 
        map_h, map_w = int(height * scale_factor), int(width * scale_factor)
        
        # Safety check for very small images
        if map_h < 1 or map_w < 1:
            map_h, map_w = height, width
            scale_factor = 1.0

        prob_map = np.zeros((map_h, map_w), dtype=np.float32)

        # Map predictions to the grid
        for pred in predictions:
            px = int(pred.x * scale_factor)
            py = int(pred.y * scale_factor)
            pw = int(pred.width * scale_factor)
            ph = int(pred.height * scale_factor)
            
            # Use 'maximum' projection to keep hotspots visible
            roi = prob_map[py:py+ph, px:px+pw]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                prob_map[py:py+ph, px:px+pw] = np.maximum(roi, pred.cancer_probability)

        # 2. Smart Smoothing
        if self.smooth_sigma > 0:
            prob_map = gaussian_filter(prob_map, sigma=self.smooth_sigma * scale_factor)

        # 3. Upscale to full resolution
        full_prob_map = cv2.resize(prob_map, (width, height), interpolation=cv2.INTER_LANCZOS4)
        full_prob_map = np.clip(full_prob_map, 0, 1)

        # 4. Generate Color Map
        # Normalize relative to max probability to make even weak signals visible
        max_val = np.max(full_prob_map)
        if max_val > 0:
            norm_map = full_prob_map / max_val
        else:
            norm_map = full_prob_map
        
        # Apply colormap (Jet or Turbo are best for visibility)
        colormap_int = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(colormap_int, cv2.COLOR_BGR2RGB)

        # 5. Adaptive Blending (The "Precision" Fix)
        # Calculate opacity mask: Transparent for low prob, Opaque for high prob
        mask = full_prob_map
        mask = np.power(mask, 1.5) # Non-linear: suppresses background noise
        mask = np.clip(mask, 0, 0.7) # Max opacity 70%
        
        mask = np.stack([mask, mask, mask], axis=2)
        
        # Blend: Original * (1-mask) + Heatmap * mask
        overlay = (img_np * (1 - mask) + heatmap_rgb * mask).astype(np.uint8)

        # 6. Calculate Stats
        stats = {
            'mean_probability': float(np.mean(full_prob_map)),
            'max_probability': float(np.max(full_prob_map)),
            'high_risk_fraction': float(np.mean(full_prob_map > 0.7)),
            'moderate_risk_fraction': float(np.mean((full_prob_map > 0.3) & (full_prob_map <= 0.7))),
            'low_risk_fraction': float(np.mean(full_prob_map <= 0.3))
        }

        return Image.fromarray(overlay), full_prob_map, stats

# --- FIX: Add Backward Compatibility Wrapper ---
# This function allows imports from __init__.py to work without crashing
def generate_heatmap_overlay(image, predictions):
    """
    Legacy wrapper for backward compatibility.
    """
    generator = HeatmapGenerator()
    overlay, _, _ = generator.generate(image, predictions)
    return overlay