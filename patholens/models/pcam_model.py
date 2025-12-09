import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import numpy as np
import os
from PIL import Image
import cv2

# Try importing torchstain for normalization, gracefully handle if missing
try:
    import torchstain
    HAS_TORCHSTAIN = True
except ImportError:
    HAS_TORCHSTAIN = False
    print("⚠️ 'torchstain' not found. Install it for better color consistency.")

def load_pcam_model(device):
    """
    Loads the ResNet18 model. 
    Prioritizes 'patholens_model.pth' (PathMNIST trained), 
    falls back to ImageNet weights.
    """
    # 1. Setup Architecture
    # We use ResNet18 as the backbone
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    
    # PathMNIST has 9 classes. We match that architecture.
    model.fc = nn.Linear(num_ftrs, 9) 
    
    # 2. Load Trained Weights
    weights_path = os.path.join(os.path.dirname(__file__), "patholens_model.pth")
    
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded trained weights from {weights_path}")
        except Exception as e:
            print(f"⚠️ Error loading weights: {e}")
    else:
        print("⚠️ No trained weights found! Run 'python train_detector.py' first. Using random weights for demo.")
    
    model.to(device)
    model.eval()
    return PCamPredictor(model, device)


class PCamPredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        
        # PathMNIST mapping: Class 8 is 'Colorectal Adenocarcinoma Epithelium'
        self.tumour_label = 8 

        # Initialize Macenko Normalizer if available
        self.normalizer = None
        if HAS_TORCHSTAIN:
            try:
                # Standardize to a reference profile
                self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
            except Exception as e:
                print(f"⚠️ Could not init Macenko normalizer: {e}")

    def preprocess_patch(self, image_pil):
        """
        Apply stain normalization to force the patch to look like the training data.
        Returns PIL Image.
        """
        if self.normalizer is None:
            return image_pil
            
        try:
            # torchstain expects numpy array
            img_np = np.array(image_pil)
            img_tensor = torch.from_numpy(img_np)
            
            # Normalize returns (norm_img, H_channel, E_channel)
            norm_img, _, _ = self.normalizer.normalize(I=img_tensor, stains=True)
            
            # Convert back to PIL
            return Image.fromarray(norm_img.numpy().astype(np.uint8))
        except Exception:
            # Fallback for whitespace/background patches where stain extraction fails
            return image_pil

    def predict_batch(self, batch_images):
        if not batch_images:
            return []

        # 1. Stain Normalization
        processed_imgs = [self.preprocess_patch(img) for img in batch_images]

        # 2. Preprocessing (PathMNIST Stats: mean=.5, std=.5)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).to(self.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).to(self.device)
        
        tensors = [torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in processed_imgs]
        inputs = torch.stack(tensors).to(self.device)
        inputs = (inputs - mean) / std

        # 3. Prediction with AMP (Speed) & TTA (Accuracy)
        # Determine device type for autocast
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        
        with torch.amp.autocast(device_type=device_type, enabled=(device_type=='cuda')):
            with torch.no_grad():
                # Original
                logits_1 = self.model(inputs)
                
                # Horizontal Flip (Test-Time Augmentation)
                logits_2 = self.model(torch.flip(inputs, [3])) 
                
                # Average predictions
                avg_logits = (logits_1 + logits_2) / 2.0
                probs = self.softmax(avg_logits)

        # 4. Format Results
        results = []
        cpu_probs = probs.cpu().float().numpy()
        
        for prob in cpu_probs:
            # Get probability of Class 8 (Tumor)
            tumour_prob = float(prob[self.tumour_label])
            
            # Binary decision: Tumor vs Everything else
            pred_class = 1 if tumour_prob > 0.5 else 0
            results.append((tumour_prob, pred_class))

        return results

# Alias for backward compatibility
class PCamClassifier(PCamPredictor):
    pass