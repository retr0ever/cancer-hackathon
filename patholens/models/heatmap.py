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
    def __init__(self, alpha=0.5, smooth_sigma=5.0, colormap='jet'):
        self.alpha = alpha
        self.smooth_sigma = smooth_sigma
        self.colormap = colormap

    def _create_overlay(self, img_np, prob_map, boost=False):
        """Helper to create a blended overlay from a probability map."""
        
        # 1. Normalize for Color
        if boost:
            # Relative Mode: Stretch 0..Max -> 0..1
            max_val = np.max(prob_map)
            norm_map = prob_map / max_val if max_val > 0.01 else prob_map
            
            # Aggressive boost: Shift colors towards red
            norm_map = np.power(norm_map, 0.7)
        else:
            # Absolute Mode: Strict 0..1
            norm_map = prob_map

        # 2. Apply Color Map
        heatmap_int = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_int, cv2.COLOR_BGR2RGB)

        # 3. Create Opacity Mask
        # We use the displayed intensity (norm_map) to determine opacity
        mask = norm_map
        if boost:
            # In relative mode, make it pop more (lower power = higher opacity for mid-range)
            mask = np.power(mask, 0.6) 
        
        # Cap opacity
        mask = np.clip(mask * 2.0, 0, self.alpha)
        mask = np.stack([mask, mask, mask], axis=2)

        # 4. Blend
        overlay = (img_np * (1 - mask) + heatmap_rgb * mask).astype(np.uint8)
        return Image.fromarray(overlay)

    def generate(self, image, predictions):
        """
        Generates both Absolute and Relative heatmaps.
        Returns: (overlay_abs, overlay_rel, raw_map, stats)
        """
        img_np = np.array(image)
        height, width = img_np.shape[:2]

        # Handle empty predictions
        if not predictions:
            empty = np.zeros((height, width))
            stats = {k: 0.0 for k in ['mean_probability', 'max_probability', 'high_risk_fraction', 'moderate_risk_fraction', 'low_risk_fraction']}
            return image, image, empty, stats

        # 1. Create Base Probability Map
        scale_factor = 0.125
        map_h, map_w = int(height * scale_factor), int(width * scale_factor)
        if map_h < 1: map_h, map_w, scale_factor = height, width, 1.0

        raw_map = np.zeros((map_h, map_w), dtype=np.float32)

        for pred in predictions:
            px, py = int(pred.x * scale_factor), int(pred.y * scale_factor)
            pw, ph = int(pred.width * scale_factor), int(pred.height * scale_factor)
            
            roi = raw_map[py:py+ph, px:px+pw]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                raw_map[py:py+ph, px:px+pw] = np.maximum(roi, pred.cancer_probability)

        # 2. Smooth and Upscale
        if self.smooth_sigma > 0:
            raw_map = gaussian_filter(raw_map, sigma=self.smooth_sigma)
        
        full_prob_map = cv2.resize(raw_map, (width, height), interpolation=cv2.INTER_LINEAR)
        full_prob_map = np.clip(full_prob_map, 0, 1)

        # 3. Generate Both Overlays
        overlay_abs = self._create_overlay(img_np, full_prob_map, boost=False)
        overlay_rel = self._create_overlay(img_np, full_prob_map, boost=True)

        # 4. Statistics
        stats = {
            'mean_probability': float(np.mean(full_prob_map)),
            'max_probability': float(np.max(full_prob_map)),
            'high_risk_fraction': float(np.mean(full_prob_map > 0.7)),
            'moderate_risk_fraction': float(np.mean((full_prob_map > 0.3) & (full_prob_map <= 0.7))),
            'low_risk_fraction': float(np.mean(full_prob_map <= 0.3))
        }

        return overlay_abs, overlay_rel, full_prob_map, stats

# Legacy wrapper fix
def generate_heatmap_overlay(image, predictions):
    generator = HeatmapGenerator()
    overlay, _, _, _ = generator.generate(image, predictions)
    return overlay