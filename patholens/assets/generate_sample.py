"""
Generate a synthetic H&E-like sample image for PathoLens demonstration.

This creates a placeholder image that mimics the appearance of H&E stained
histopathology tissue. Replace with real sample images for actual testing.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random


def generate_sample_histo(
    width: int = 1024,
    height: int = 1024,
    seed: int = 42
) -> Image.Image:
    """
    Generate a synthetic H&E-like histopathology image.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for reproducibility

    Returns:
        PIL Image with synthetic tissue appearance
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create base image with pinkish background (eosin-like)
    base_color = np.array([245, 220, 225])  # Light pink background
    image = np.ones((height, width, 3), dtype=np.uint8) * base_color

    # Add tissue texture variation
    noise = np.random.normal(0, 15, (height, width, 3))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # Convert to PIL for drawing
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    # Generate nuclei (blue/purple spots representing haematoxylin staining)
    num_nuclei = random.randint(800, 1200)

    # Nuclei colors (haematoxylin - blue/purple range)
    nuclei_colors = [
        (80, 60, 130),   # Deep purple
        (100, 80, 150),  # Medium purple
        (70, 50, 120),   # Dark purple
        (90, 70, 140),   # Purple
        (60, 40, 100),   # Very dark purple
    ]

    for _ in range(num_nuclei):
        x = random.randint(20, width - 20)
        y = random.randint(20, height - 20)

        # Varied nucleus sizes
        radius = random.randint(4, 12)

        # Random color from palette
        color = random.choice(nuclei_colors)

        # Add some color variation
        color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in color)

        # Draw nucleus (slightly elliptical)
        eccentricity = random.uniform(0.7, 1.3)
        rx = int(radius * eccentricity)
        ry = int(radius / eccentricity)

        draw.ellipse(
            [x - rx, y - ry, x + rx, y + ry],
            fill=color
        )

    # Add some larger structures (simulating tissue architecture)
    num_structures = random.randint(5, 10)
    structure_colors = [
        (230, 200, 210),  # Light pink
        (240, 215, 220),  # Very light pink
        (220, 190, 200),  # Medium pink
    ]

    for _ in range(num_structures):
        cx = random.randint(100, width - 100)
        cy = random.randint(100, height - 100)
        radius = random.randint(50, 150)

        color = random.choice(structure_colors)

        # Draw irregular structure
        points = []
        num_points = random.randint(6, 12)
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            r = radius * random.uniform(0.7, 1.3)
            px = cx + int(r * np.cos(angle))
            py = cy + int(r * np.sin(angle))
            points.append((px, py))

        if len(points) >= 3:
            draw.polygon(points, fill=color)

    # Add some stromal tissue (fibrous areas)
    num_stromal = random.randint(3, 7)
    for _ in range(num_stromal):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = x1 + random.randint(-200, 200)
        y2 = y1 + random.randint(-200, 200)

        color = (235, 210, 215, 100)  # Semi-transparent pink

        for offset in range(-3, 4):
            draw.line(
                [(x1, y1 + offset), (x2, y2 + offset)],
                fill=(235, 210, 215),
                width=2
            )

    # Apply slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Add a "cancerous region" in one area (darker, denser nuclei)
    # This creates a region that the classifier might flag
    cancer_x = random.randint(200, width - 300)
    cancer_y = random.randint(200, height - 300)
    cancer_radius = random.randint(100, 200)

    draw = ImageDraw.Draw(img)

    # Denser nuclei in "suspicious" region
    for _ in range(200):
        angle = random.uniform(0, 2 * np.pi)
        r = random.uniform(0, cancer_radius)
        x = int(cancer_x + r * np.cos(angle))
        y = int(cancer_y + r * np.sin(angle))

        # Larger, more irregular nuclei (suggesting malignancy)
        radius = random.randint(6, 15)

        # Darker, more irregular colors
        color = (
            random.randint(50, 80),
            random.randint(30, 60),
            random.randint(90, 130)
        )

        # More irregular shapes
        rx = int(radius * random.uniform(0.5, 1.5))
        ry = int(radius * random.uniform(0.5, 1.5))

        draw.ellipse(
            [x - rx, y - ry, x + rx, y + ry],
            fill=color
        )

    return img


if __name__ == "__main__":
    print("Generating synthetic H&E sample image...")
    img = generate_sample_histo()
    img.save("sample_histo.png")
    print("Saved to sample_histo.png")
    print("\nNote: This is a synthetic placeholder image.")
    print("For realistic testing, replace with actual histopathology images.")
