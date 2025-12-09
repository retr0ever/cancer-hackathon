"""
Nuclei Segmentation Module for PathoLens

Implements colour deconvolution and morphological operations for
nuclei detection in H&E stained histopathology images.
"""

import numpy as np
import cv2
from PIL import Image
from skimage import morphology, measure, filters, exposure
from skimage.color import rgb2gray, rgb2hed
from skimage.segmentation import watershed
from scipy import ndimage
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class NucleusInfo:
    """Information about a detected nucleus."""
    centroid: Tuple[float, float]
    area: float
    perimeter: float
    eccentricity: float
    solidity: float
    mean_intensity: float
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray


class NucleiSegmenter:
    """
    Nuclei segmentation using colour deconvolution and morphological operations.
    Optimised for H&E stained tissue sections.
    """

    # Stain vectors for H&E colour deconvolution
    # These are approximate values for standard H&E staining
    HE_STAIN_MATRIX = np.array([
        [0.65, 0.70, 0.29],   # Haematoxylin
        [0.07, 0.99, 0.11],   # Eosin
        [0.27, 0.57, 0.78]    # DAB (background)
    ])

    def __init__(
        self,
        min_nucleus_area: int = 50,
        max_nucleus_area: int = 5000,
        threshold_method: str = 'otsu',
        use_watershed: bool = True
    ):
        """
        Initialise the nuclei segmenter.

        Args:
            min_nucleus_area: Minimum area in pixels for valid nucleus
            max_nucleus_area: Maximum area in pixels for valid nucleus
            threshold_method: Thresholding method ('otsu', 'adaptive', 'li')
            use_watershed: Whether to use watershed for separating touching nuclei
        """
        self.min_nucleus_area = min_nucleus_area
        self.max_nucleus_area = max_nucleus_area
        self.threshold_method = threshold_method
        self.use_watershed = use_watershed

    def colour_deconvolution(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate H&E stains using colour deconvolution.

        Args:
            image: RGB image as numpy array

        Returns:
            Tuple of (haematoxylin_channel, eosin_channel)
        """
        # Use skimage's built-in H&E deconvolution
        hed = rgb2hed(image)

        haematoxylin = hed[:, :, 0]
        eosin = hed[:, :, 1]

        # Normalise to 0-255 range
        haematoxylin = exposure.rescale_intensity(haematoxylin, out_range=(0, 255)).astype(np.uint8)
        eosin = exposure.rescale_intensity(eosin, out_range=(0, 255)).astype(np.uint8)

        return haematoxylin, eosin

    def threshold_nuclei(self, haematoxylin: np.ndarray) -> np.ndarray:
        """
        Apply thresholding to segment nuclei from haematoxylin channel.

        Args:
            haematoxylin: Haematoxylin channel image

        Returns:
            Binary mask of nuclei
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(haematoxylin, (5, 5), 0)

        if self.threshold_method == 'otsu':
            threshold = filters.threshold_otsu(blurred)
            binary = blurred > threshold
        elif self.threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            ) > 0
        elif self.threshold_method == 'li':
            threshold = filters.threshold_li(blurred)
            binary = blurred > threshold
        else:
            threshold = filters.threshold_otsu(blurred)
            binary = blurred > threshold

        return binary.astype(np.uint8)

    def morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the binary mask.

        Args:
            binary: Binary mask

        Returns:
            Cleaned binary mask
        """
        # Remove small objects (noise)
        cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=20)

        # Fill holes in nuclei
        cleaned = ndimage.binary_fill_holes(cleaned)

        # Opening to separate slightly touching nuclei
        kernel = morphology.disk(2)
        cleaned = morphology.opening(cleaned, kernel)

        # Closing to smooth boundaries
        cleaned = morphology.closing(cleaned, kernel)

        return cleaned.astype(np.uint8)

    def watershed_separation(
        self,
        binary: np.ndarray,
        haematoxylin: np.ndarray
    ) -> np.ndarray:
        """
        Use watershed algorithm to separate touching nuclei.

        Args:
            binary: Binary mask of nuclei
            haematoxylin: Haematoxylin channel for distance weighting

        Returns:
            Labelled image with separated nuclei
        """
        # Distance transform
        distance = ndimage.distance_transform_edt(binary)

        # Find local maxima as markers
        local_max_coords = morphology.local_maxima(distance)
        markers = measure.label(local_max_coords)

        # Apply watershed
        labels = watershed(-distance, markers, mask=binary)

        return labels

    def extract_nucleus_features(
        self,
        labels: np.ndarray,
        original_image: np.ndarray,
        haematoxylin: np.ndarray
    ) -> List[NucleusInfo]:
        """
        Extract features for each detected nucleus.

        Args:
            labels: Labelled nuclei image
            original_image: Original RGB image
            haematoxylin: Haematoxylin channel

        Returns:
            List of NucleusInfo objects
        """
        nuclei = []
        props = measure.regionprops(labels, intensity_image=haematoxylin)

        for prop in props:
            # Filter by area
            if prop.area < self.min_nucleus_area or prop.area > self.max_nucleus_area:
                continue

            # Create mask for this nucleus
            mask = (labels == prop.label).astype(np.uint8)

            nucleus = NucleusInfo(
                centroid=(prop.centroid[1], prop.centroid[0]),  # x, y
                area=prop.area,
                perimeter=prop.perimeter,
                eccentricity=prop.eccentricity,
                solidity=prop.solidity,
                mean_intensity=prop.mean_intensity,
                bbox=prop.bbox,
                mask=mask
            )
            nuclei.append(nucleus)

        return nuclei

    def segment(self, image: Image.Image) -> Tuple[List[NucleusInfo], np.ndarray, np.ndarray]:
        """
        Perform full nuclei segmentation pipeline.

        Args:
            image: PIL Image of the tissue section

        Returns:
            Tuple of (nuclei_list, labelled_mask, haematoxylin_channel)
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Ensure RGB format
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Colour deconvolution
        haematoxylin, eosin = self.colour_deconvolution(img_array)

        # Threshold nuclei
        binary = self.threshold_nuclei(haematoxylin)

        # Morphological cleanup
        cleaned = self.morphological_cleanup(binary)

        # Separate touching nuclei
        if self.use_watershed:
            labels = self.watershed_separation(cleaned, haematoxylin)
        else:
            labels = measure.label(cleaned)

        # Extract features
        nuclei = self.extract_nucleus_features(labels, img_array, haematoxylin)

        return nuclei, labels, haematoxylin


def segment_nuclei(
    image: Image.Image,
    min_area: int = 50,
    max_area: int = 5000
) -> Tuple[List[NucleusInfo], np.ndarray]:
    """
    Convenience function for nuclei segmentation.

    Args:
        image: PIL Image
        min_area: Minimum nucleus area
        max_area: Maximum nucleus area

    Returns:
        Tuple of (nuclei_list, labelled_mask)
    """
    segmenter = NucleiSegmenter(
        min_nucleus_area=min_area,
        max_nucleus_area=max_area
    )
    nuclei, labels, _ = segmenter.segment(image)
    return nuclei, labels


def compute_nuclei_statistics(nuclei: List[NucleusInfo]) -> Dict:
    """
    Compute aggregate statistics for detected nuclei.

    Args:
        nuclei: List of detected nuclei

    Returns:
        Dictionary of statistics
    """
    if not nuclei:
        return {
            'count': 0,
            'mean_area': 0,
            'std_area': 0,
            'mean_eccentricity': 0,
            'mean_solidity': 0,
            'area_variation': 0
        }

    areas = [n.area for n in nuclei]
    eccentricities = [n.eccentricity for n in nuclei]
    solidities = [n.solidity for n in nuclei]

    return {
        'count': len(nuclei),
        'mean_area': np.mean(areas),
        'std_area': np.std(areas),
        'mean_eccentricity': np.mean(eccentricities),
        'mean_solidity': np.mean(solidities),
        'area_variation': np.std(areas) / (np.mean(areas) + 1e-8)  # Coefficient of variation
    }
