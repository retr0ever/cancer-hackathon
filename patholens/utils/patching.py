"""
Patch Extraction Module for PathoLens

Handles extraction of overlapping patches from whole slide images
for processing by the cancer classifier.
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Generator
from dataclasses import dataclass
import cv2


@dataclass
class Patch:
    """Represents an extracted image patch."""
    image: Image.Image
    x: int
    y: int
    width: int
    height: int
    tissue_fraction: float


class PatchExtractor:
    """
    Extracts overlapping patches from histopathology images.
    Includes tissue detection to avoid processing empty regions.
    """

    def __init__(
        self,
        patch_size: int = 224,
        stride: int = 112,
        min_tissue_fraction: float = 0.5,
        tissue_threshold: int = 220
    ):
        """
        Initialise the patch extractor.

        Args:
            patch_size: Size of square patches to extract
            stride: Step size between patches (< patch_size for overlap)
            min_tissue_fraction: Minimum fraction of tissue in patch
            tissue_threshold: Grayscale threshold for tissue detection
        """
        self.patch_size = patch_size
        self.stride = stride
        self.min_tissue_fraction = min_tissue_fraction
        self.tissue_threshold = tissue_threshold

    def detect_tissue_mask(self, image: Image.Image) -> np.ndarray:
        """
        Create a binary mask indicating tissue regions.

        H&E stained tissue typically appears pink/purple against
        a white background. We use grayscale thresholding to
        detect tissue regions.

        Args:
            image: PIL Image of the slide

        Returns:
            Binary mask (1 = tissue, 0 = background)
        """
        # Convert to grayscale
        gray = np.array(image.convert('L'))

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Threshold - tissue is darker than background
        tissue_mask = blurred < self.tissue_threshold

        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        tissue_mask = cv2.morphologyEx(
            tissue_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        )
        tissue_mask = cv2.morphologyEx(
            tissue_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        return tissue_mask

    def compute_tissue_fraction(
        self,
        tissue_mask: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> float:
        """
        Compute the fraction of tissue in a patch region.

        Args:
            tissue_mask: Binary tissue mask
            x, y: Top-left corner of patch
            width, height: Patch dimensions

        Returns:
            Fraction of pixels that are tissue (0-1)
        """
        patch_mask = tissue_mask[y:y+height, x:x+width]
        if patch_mask.size == 0:
            return 0.0
        return np.mean(patch_mask)

    def extract_patches(
        self,
        image: Image.Image,
        filter_tissue: bool = True
    ) -> List[Patch]:
        """
        Extract all patches from an image.

        Args:
            image: PIL Image to extract patches from
            filter_tissue: Whether to filter out non-tissue patches

        Returns:
            List of Patch objects
        """
        width, height = image.size
        patches = []

        # Create tissue mask if filtering
        if filter_tissue:
            tissue_mask = self.detect_tissue_mask(image)
        else:
            tissue_mask = np.ones((height, width), dtype=np.uint8)

        # Extract patches
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                # Check tissue fraction
                tissue_frac = self.compute_tissue_fraction(
                    tissue_mask, x, y, self.patch_size, self.patch_size
                )

                if tissue_frac >= self.min_tissue_fraction or not filter_tissue:
                    # Extract patch
                    patch_img = image.crop((
                        x, y,
                        x + self.patch_size,
                        y + self.patch_size
                    ))

                    patches.append(Patch(
                        image=patch_img,
                        x=x,
                        y=y,
                        width=self.patch_size,
                        height=self.patch_size,
                        tissue_fraction=tissue_frac
                    ))

        return patches

    def extract_patches_generator(
        self,
        image: Image.Image,
        filter_tissue: bool = True
    ) -> Generator[Patch, None, None]:
        """
        Generator version of patch extraction for memory efficiency.

        Args:
            image: PIL Image to extract patches from
            filter_tissue: Whether to filter out non-tissue patches

        Yields:
            Patch objects one at a time
        """
        width, height = image.size

        if filter_tissue:
            tissue_mask = self.detect_tissue_mask(image)
        else:
            tissue_mask = np.ones((height, width), dtype=np.uint8)

        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                tissue_frac = self.compute_tissue_fraction(
                    tissue_mask, x, y, self.patch_size, self.patch_size
                )

                if tissue_frac >= self.min_tissue_fraction or not filter_tissue:
                    patch_img = image.crop((
                        x, y,
                        x + self.patch_size,
                        y + self.patch_size
                    ))

                    yield Patch(
                        image=patch_img,
                        x=x,
                        y=y,
                        width=self.patch_size,
                        height=self.patch_size,
                        tissue_fraction=tissue_frac
                    )

    def get_grid_info(
        self,
        image_size: Tuple[int, int]
    ) -> Tuple[int, int, int]:
        """
        Calculate grid information for patch extraction.

        Args:
            image_size: (width, height) of image

        Returns:
            Tuple of (n_patches_x, n_patches_y, total_patches)
        """
        width, height = image_size

        n_patches_x = (width - self.patch_size) // self.stride + 1
        n_patches_y = (height - self.patch_size) // self.stride + 1

        return n_patches_x, n_patches_y, n_patches_x * n_patches_y


def extract_patches(
    image: Image.Image,
    patch_size: int = 224,
    stride: int = 112,
    min_tissue: float = 0.5
) -> List[Patch]:
    """
    Convenience function to extract patches from an image.

    Args:
        image: PIL Image
        patch_size: Size of patches
        stride: Step between patches
        min_tissue: Minimum tissue fraction

    Returns:
        List of Patch objects
    """
    extractor = PatchExtractor(
        patch_size=patch_size,
        stride=stride,
        min_tissue_fraction=min_tissue
    )
    return extractor.extract_patches(image)


def visualise_patches(
    image: Image.Image,
    patches: List[Patch],
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2
) -> Image.Image:
    """
    Visualise extracted patches on the original image.

    Args:
        image: Original PIL Image
        patches: List of extracted patches
        line_color: RGB colour for patch boundaries
        line_width: Width of boundary lines

    Returns:
        PIL Image with patch boundaries drawn
    """
    img_array = np.array(image.copy())

    for patch in patches:
        cv2.rectangle(
            img_array,
            (patch.x, patch.y),
            (patch.x + patch.width, patch.y + patch.height),
            line_color,
            line_width
        )

    return Image.fromarray(img_array)
