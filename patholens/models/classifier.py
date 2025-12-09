"""
Cancer Classification Module for PathoLens

Provides patch-level cancer probability prediction using a pretrained
ResNet18 model. Includes Grad-CAM support for interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for model interpretability.
    Highlights regions that contribute most to the classification decision.
    """

    def __init__(self, model: nn.Module, target_layer: str = 'layer4'):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Get target layer
        target = dict(self.model.named_modules())[self.target_layer]
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input.

        Args:
            input_tensor: Preprocessed input image tensor
            target_class: Class index to generate CAM for (None = predicted class)

        Returns:
            Normalised heatmap as numpy array
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalise
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


class CancerClassifier:
    """
    Wrapper class for cancer classification with preprocessing and inference.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.grad_cam = GradCAM(model, target_layer='layer4')

        # Standard preprocessing for histopathology patches
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]     # ImageNet stds
            )
        ])

        # Histopathology-specific normalisation (can be enabled for better results)
        self.histo_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7, 0.55, 0.7],  # Typical H&E stain values
                std=[0.15, 0.15, 0.15]
            )
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image for model input."""
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[float, int]:
        """
        Predict cancer probability for a single patch.

        Args:
            image: PIL Image of the patch

        Returns:
            Tuple of (cancer_probability, predicted_class)
        """
        input_tensor = self.preprocess(image)
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)

        cancer_prob = probabilities[0, 1].item()  # Index 1 = cancer class
        predicted_class = output.argmax(dim=1).item()

        return cancer_prob, predicted_class

    def predict_with_cam(
        self,
        image: Image.Image
    ) -> Tuple[float, int, np.ndarray]:
        """
        Predict with Grad-CAM visualisation.

        Returns:
            Tuple of (cancer_probability, predicted_class, grad_cam_heatmap)
        """
        input_tensor = self.preprocess(image)

        # Enable gradients for CAM
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)

        cancer_prob = probabilities[0, 1].item()
        predicted_class = output.argmax(dim=1).item()

        # Generate Grad-CAM for cancer class
        cam = self.grad_cam.generate(input_tensor, target_class=1)

        return cancer_prob, predicted_class, cam

    @torch.no_grad()
    def predict_batch(self, images: List[Image.Image]) -> List[Tuple[float, int]]:
        """
        Predict cancer probability for a batch of patches.

        Args:
            images: List of PIL Images

        Returns:
            List of (cancer_probability, predicted_class) tuples
        """
        if not images:
            return []

        # Batch preprocessing
        tensors = torch.stack([self.transform(img) for img in images])
        tensors = tensors.to(self.device)

        output = self.model(tensors)
        probabilities = F.softmax(output, dim=1)

        results = []
        for i in range(len(images)):
            cancer_prob = probabilities[i, 1].item()
            predicted_class = output[i].argmax().item()
            results.append((cancer_prob, predicted_class))

        return results


def predict_patch(
    model: nn.Module,
    patch: Image.Image,
    device: torch.device
) -> Tuple[float, int]:
    """
    Convenience function for single patch prediction.

    Args:
        model: Loaded cancer classifier model
        patch: PIL Image of the patch
        device: Torch device

    Returns:
        Tuple of (cancer_probability, predicted_class)
    """
    classifier = CancerClassifier(model, device)
    return classifier.predict(patch)
