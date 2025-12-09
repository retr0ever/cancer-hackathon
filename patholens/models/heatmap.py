"""
Heatmap Generation Module for PathoLens

Creates cancer probability heatmaps by analysing patches across
the whole slide image and reconstructing a spatial probability map.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import torch


@dataclass
class PatchPrediction:
    """Stores prediction information for a single patch."""
    x: int
    y: int
    width: int
    height: int
    cancer_probability: float
    predicted_class: int
    grad_cam: Optional[np.ndarray] = None


class HeatmapGenerator:
    """
    Generates cancer probability heatmaps from patch predictions.
    """

    def __init__(
        self,
        colormap: str = 'RdYlBu_r',
        alpha: float = 0.5,
        threshold: float = 0.0,
        smooth_sigma: float = 2.0
    ):
        """
        Initialise the heatmap generator.

        Args:
            colormap: Matplotlib colormap name
            alpha: Opacity for overlay (0-1)
            threshold: Minimum probability to display
            smooth_sigma: Gaussian smoothing sigma
        """
        self.colormap = colormap
        self.alpha = alpha
        self.threshold = threshold
        self.smooth_sigma = smooth_sigma

    def create_probability_map(
        self,
        predictions: List[PatchPrediction],
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create a probability map from patch predictions.

        Args:
            predictions: List of patch predictions
            original_size: (width, height) of original image

        Returns:
            2D probability map
        """
        width, height = original_size
        prob_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)

        for pred in predictions:
            # Ensure coordinates are within bounds
            x_end = min(pred.x + pred.width, width)
            y_end = min(pred.y + pred.height, height)

            # Add probability to the region
            prob_map[pred.y:y_end, pred.x:x_end] += pred.cancer_probability
            count_map[pred.y:y_end, pred.x:x_end] += 1

        # Average overlapping regions
        count_map[count_map == 0] = 1  # Avoid division by zero
        prob_map = prob_map / count_map

        return prob_map

    def smooth_heatmap(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to the probability map.

        Args:
            prob_map: Raw probability map

        Returns:
            Smoothed probability map
        """
        if self.smooth_sigma > 0:
            kernel_size = int(self.smooth_sigma * 6) | 1  # Ensure odd
            smoothed = cv2.GaussianBlur(
                prob_map,
                (kernel_size, kernel_size),
                self.smooth_sigma
            )
            return smoothed
        return prob_map

    def apply_colormap(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Apply a colormap to the probability map.

        Args:
            prob_map: Probability map (0-1 values)

        Returns:
            RGB coloured heatmap
        """
        # Get colormap
        cmap = plt.get_cmap(self.colormap)

        # Apply threshold
        masked_map = prob_map.copy()
        masked_map[masked_map < self.threshold] = 0

        # Normalise to 0-1
        if masked_map.max() > 0:
            normalised = masked_map / masked_map.max()
        else:
            normalised = masked_map

        # Apply colormap
        coloured = cmap(normalised)

        # Convert to RGB (0-255)
        rgb = (coloured[:, :, :3] * 255).astype(np.uint8)

        return rgb

    def create_overlay(
        self,
        original_image: Image.Image,
        prob_map: np.ndarray
    ) -> Image.Image:
        """
        Create an overlay of the heatmap on the original image.

        Args:
            original_image: Original PIL Image
            prob_map: Probability map

        Returns:
            PIL Image with heatmap overlay
        """
        # Convert original to numpy
        original_array = np.array(original_image)
        if len(original_array.shape) == 2:
            original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
        elif original_array.shape[2] == 4:
            original_array = cv2.cvtColor(original_array, cv2.COLOR_RGBA2RGB)

        # Smooth the probability map
        smoothed = self.smooth_heatmap(prob_map)

        # Apply colormap
        heatmap_rgb = self.apply_colormap(smoothed)

        # Create alpha mask based on probability
        alpha_mask = smoothed.copy()
        alpha_mask[alpha_mask < self.threshold] = 0
        alpha_mask = (alpha_mask * self.alpha)[:, :, np.newaxis]

        # Blend images
        overlay = (
            original_array * (1 - alpha_mask) +
            heatmap_rgb * alpha_mask
        ).astype(np.uint8)

        return Image.fromarray(overlay)

    def create_legend(
        self,
        figsize: Tuple[int, int] = (2, 6)
    ) -> Image.Image:
        """
        Create a colour legend for the heatmap.

        Returns:
            PIL Image of the legend
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create gradient
        gradient = np.linspace(0, 1, 256).reshape(-1, 1)

        # Display gradient with colormap
        cmap = plt.get_cmap(self.colormap)
        ax.imshow(gradient, aspect='auto', cmap=cmap, origin='lower')

        # Configure axis
        ax.set_ylabel('Cancer Probability', fontsize=10, fontfamily='sans-serif')
        ax.set_yticks([0, 128, 255])
        ax.set_yticklabels(['0.0', '0.5', '1.0'])
        ax.set_xticks([])

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Convert to PIL Image
        fig.tight_layout()
        fig.canvas.draw()

        # Get image from figure
        width, height = fig.canvas.get_width_height()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape((height, width, 3))

        plt.close(fig)

        return Image.fromarray(img_array)

    def generate(
        self,
        original_image: Image.Image,
        predictions: List[PatchPrediction]
    ) -> Tuple[Image.Image, np.ndarray, Dict]:
        """
        Generate complete heatmap analysis.

        Args:
            original_image: Original slide image
            predictions: List of patch predictions

        Returns:
            Tuple of (overlay_image, probability_map, statistics)
        """
        # Create probability map
        prob_map = self.create_probability_map(
            predictions,
            (original_image.width, original_image.height)
        )

        # Create overlay
        overlay = self.create_overlay(original_image, prob_map)

        # Compute statistics
        stats = self.compute_statistics(predictions, prob_map)

        return overlay, prob_map, stats

    def compute_statistics(
        self,
        predictions: List[PatchPrediction],
        prob_map: np.ndarray
    ) -> Dict:
        """
        Compute heatmap statistics.

        Args:
            predictions: List of patch predictions
            prob_map: Generated probability map

        Returns:
            Dictionary of statistics
        """
        probs = [p.cancer_probability for p in predictions]

        return {
            'mean_probability': np.mean(probs) if probs else 0,
            'max_probability': np.max(probs) if probs else 0,
            'min_probability': np.min(probs) if probs else 0,
            'std_probability': np.std(probs) if probs else 0,
            'high_risk_fraction': np.mean(prob_map > 0.7) if prob_map.size > 0 else 0,
            'moderate_risk_fraction': np.mean((prob_map > 0.3) & (prob_map <= 0.7)) if prob_map.size > 0 else 0,
            'low_risk_fraction': np.mean(prob_map <= 0.3) if prob_map.size > 0 else 0,
            'num_patches': len(predictions)
        }


def generate_heatmap_overlay(
    original_image: Image.Image,
    predictions: List[PatchPrediction],
    alpha: float = 0.5,
    colormap: str = 'RdYlBu_r'
) -> Tuple[Image.Image, np.ndarray]:
    """
    Convenience function to generate a heatmap overlay.

    Args:
        original_image: Original slide image
        predictions: List of patch predictions
        alpha: Overlay opacity
        colormap: Matplotlib colormap name

    Returns:
        Tuple of (overlay_image, probability_map)
    """
    generator = HeatmapGenerator(colormap=colormap, alpha=alpha)
    overlay, prob_map, _ = generator.generate(original_image, predictions)
    return overlay, prob_map
