"""
PCam (PatchCamelyon) Model for PathoLens

Provides a ResNet18-based classifier optimised for 96x96 PCam patches.
Uses pretrained ImageNet weights with adapted architecture for histopathology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path


class PCamClassifier(nn.Module):
    """
    ResNet18-based classifier for PCam 96x96 patches.

    The model is adapted for binary classification (tumour/no tumour)
    with the central 32x32 region determining the label.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load ResNet18 backbone
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)

        # Modify first conv for potentially smaller inputs
        # Keep original for 96x96 (works fine with standard ResNet)

        # Replace classifier head for binary classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of tumour class."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)[:, 1]


class PCamPredictor:
    """
    High-level predictor for PCam-style histopathology patches.

    Handles preprocessing, inference, and probability extraction.
    """

    # PCam normalisation values (computed from dataset)
    PCAM_MEAN = [0.701, 0.538, 0.692]
    PCAM_STD = [0.235, 0.277, 0.213]

    # ImageNet normalisation (fallback)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        use_pcam_norm: bool = True
    ):
        """
        Initialise the predictor.

        Args:
            model: Pre-loaded model (or None to create new)
            device: Torch device
            use_pcam_norm: Use PCam-specific normalisation
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')

        self.device = device

        if model is None:
            self.model = PCamClassifier(pretrained=True)
        else:
            self.model = model

        self.model = self.model.to(device)
        self.model.eval()

        # Setup transforms
        if use_pcam_norm:
            mean, std = self.PCAM_MEAN, self.PCAM_STD
        else:
            mean, std = self.IMAGENET_MEAN, self.IMAGENET_STD

        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[float, int]:
        """
        Predict tumour probability for a single patch.

        Args:
            image: PIL Image (will be resized to 96x96)

        Returns:
            Tuple of (tumour_probability, predicted_class)
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)

        tumour_prob = probs[0, 1].item()
        pred_class = logits.argmax(dim=1).item()

        return tumour_prob, pred_class

    @torch.no_grad()
    def predict_batch(self, images: List[Image.Image]) -> List[Tuple[float, int]]:
        """
        Predict tumour probability for a batch of patches.

        Args:
            images: List of PIL Images

        Returns:
            List of (tumour_probability, predicted_class) tuples
        """
        if not images:
            return []

        tensors = torch.stack([self.transform(img) for img in images])
        tensors = tensors.to(self.device)

        logits = self.model(tensors)
        probs = F.softmax(logits, dim=1)

        results = []
        for i in range(len(images)):
            tumour_prob = probs[i, 1].item()
            pred_class = logits[i].argmax().item()
            results.append((tumour_prob, pred_class))

        return results


def load_pcam_model(
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> PCamPredictor:
    """
    Load PCam model with optional custom weights.

    Args:
        weights_path: Path to trained weights (optional)
        device: Torch device

    Returns:
        Configured PCamPredictor
    """
    model = PCamClassifier(pretrained=True)

    if weights_path and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded PCam weights from {weights_path}")
    else:
        print("Using ImageNet pretrained weights adapted for PCam")

    return PCamPredictor(model=model, device=device)


def download_pcam_samples(output_dir: str, num_samples: int = 5) -> List[str]:
    """
    Download sample PCam images for demonstration.

    Uses torchvision's PCAM dataset to get real histopathology samples.

    Args:
        output_dir: Directory to save samples
        num_samples: Number of samples to download

    Returns:
        List of saved file paths
    """
    try:
        from torchvision.datasets import PCAM
        import h5py
    except ImportError:
        print("Install h5py for PCam dataset: pip install h5py")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    try:
        # Download test split (smallest)
        dataset = PCAM(
            root=str(output_path / 'pcam_data'),
            split='test',
            download=True
        )

        # Save sample images
        for i in range(min(num_samples, len(dataset))):
            img, label = dataset[i]
            label_str = 'tumour' if label == 1 else 'normal'
            filename = f'pcam_sample_{i}_{label_str}.png'
            filepath = output_path / filename
            img.save(filepath)
            saved_paths.append(str(filepath))

        print(f"Saved {len(saved_paths)} PCam samples to {output_dir}")

    except Exception as e:
        print(f"Could not download PCam samples: {e}")

    return saved_paths
