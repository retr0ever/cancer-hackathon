"""
Model Loading Utilities for PathoLens

Handles loading and initialisation of pretrained models for:
- Cancer classification (ResNet18-based)
- Nuclei segmentation
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import os
import ssl
import certifi


def _fix_ssl_certificates():
    """
    Fix SSL certificate issues on macOS.
    This is a common issue with Python installations on macOS.
    """
    try:
        # Try to use certifi certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl._create_default_https_context = lambda: ssl_context
    except Exception:
        # Fallback: disable SSL verification (not ideal but works for demo)
        ssl._create_default_https_context = ssl._create_unverified_context


def get_device() -> torch.device:
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_cancer_classifier(
    weights_path: Optional[str] = None,
    num_classes: int = 2,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load a pretrained ResNet18 cancer classifier.

    For the MVP, we use ImageNet pretrained weights and adapt the final layer
    for binary classification (cancer vs non-cancer). In production, this would
    be fine-tuned on histopathology datasets like BACH or Camelyon.

    Args:
        weights_path: Path to fine-tuned weights (optional)
        num_classes: Number of output classes (default: 2 for binary)
        device: Target device for the model

    Returns:
        Loaded and configured model
    """
    if device is None:
        device = get_device()

    # Try to load pretrained ResNet18 with SSL fix
    model = None
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print("Loaded ImageNet pretrained weights successfully")
    except Exception as e:
        if "SSL" in str(e) or "certificate" in str(e).lower():
            print("SSL certificate issue detected, applying fix...")
            _fix_ssl_certificates()
            try:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                print("Loaded ImageNet pretrained weights after SSL fix")
            except Exception:
                pass

    # Fallback to random weights if download fails
    if model is None:
        print("Warning: Could not download pretrained weights. Using random initialisation.")
        print("For better results, run: /Applications/Python\\ 3.12/Install\\ Certificates.command")
        model = models.resnet18(weights=None)

    # Modify final fully connected layer for cancer classification
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )

    # Load fine-tuned weights if provided
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded fine-tuned weights from {weights_path}")

    model = model.to(device)
    model.eval()

    return model


def load_segmentation_model(
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Optional[nn.Module]:
    """
    Load a nuclei segmentation model.

    For the MVP, we use a simplified approach with colour deconvolution
    and morphological operations rather than a deep learning model.
    This function provides the interface for future integration of
    models like HoVerNet or StarDist.

    Args:
        weights_path: Path to segmentation model weights
        device: Target device for the model

    Returns:
        Loaded segmentation model or None for classical approach
    """
    if device is None:
        device = get_device()

    # For MVP: Return None to indicate classical segmentation approach
    # Future: Load HoVerNet, StarDist, or similar
    if weights_path and os.path.exists(weights_path):
        # Placeholder for deep learning segmentation model
        # model = torch.load(weights_path, map_location=device)
        # return model
        pass

    print("Using classical segmentation approach (colour deconvolution + morphology)")
    return None


def get_model_info() -> dict:
    """
    Return information about loaded models for display purposes.
    """
    return {
        'classifier': {
            'architecture': 'ResNet18',
            'pretrained_on': 'ImageNet',
            'fine_tuned': False,  # Set to True when using fine-tuned weights
            'input_size': (224, 224),
            'output_classes': ['Non-cancerous', 'Cancerous']
        },
        'segmentation': {
            'method': 'Colour Deconvolution + Morphological Operations',
            'stain_vectors': 'H&E optimised',
            'deep_learning': False
        }
    }
