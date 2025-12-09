# models/pcam_model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import numpy as np
import os

def load_pcam_model(device):
    """
    Loads ResNet18. Tries to load trained weights 'patholens_model.pth'.
    If not found, falls back to ImageNet (which will perform poorly).
    """
    # 1. Setup Architecture
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    # PathMNIST has 9 classes. We load the 9-class head, 
    # then we will map specific classes to "Tumour" during prediction.
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
        print("⚠️ No trained weights found! Run 'python train_detector.py' first.")
    
    model.to(device)
    model.eval()
    return PCamPredictor(model, device)

class PCamPredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.softmax = nn.Softmax(dim=1)

        # PathMNIST Class Mapping
        # 'CRC' (Colorectal Cancer) is label 'STR' (Stroma) vs 'TUM' (Tumour) etc.
        # PathMNIST mapping: 0:Adipose, 1:Background, 2:Debris, 3:Lymphocytes, 
        # 4:Mucus, 5:Smooth Muscle, 6:Normal Colon Mucosa, 7:Cancer-Associated Stroma, 8:Colorectal Adenocarcinoma Epithelium
        # WE WANT CLASS 8 (Tumor) vs OTHERS.
        self.tumour_label = 8 

    def predict_batch(self, batch_images):
        if not batch_images:
            return []

        # Preprocessing (must match training!)
        # PathMNIST used mean=.5, std=.5
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).to(self.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).to(self.device)
        
        tensors = [torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in batch_images]
        inputs = torch.stack(tensors).to(self.device)
        inputs = (inputs - mean) / std

        with torch.no_grad():
            # Test-Time Augmentation (TTA)
            logits_1 = self.model(inputs)
            logits_2 = self.model(torch.flip(inputs, [3])) # H-Flip
            
            avg_logits = (logits_1 + logits_2) / 2.0
            probs = self.softmax(avg_logits)

        results = []
        cpu_probs = probs.cpu().numpy()
        
        for prob in cpu_probs:
            # Get probability of Class 8 (Tumor)
            # We treat everything else as "Non-Tumor"
            tumour_prob = float(prob[self.tumour_label])
            
            # If "Stroma" (Class 7) is high, it's also risky, but let's stick to strict Tumor (8)
            # You could sum prob[7] + prob[8] for a more sensitive detector.
            
            pred_class = 1 if tumour_prob > 0.5 else 0
            results.append((tumour_prob, pred_class))

        return results

# Alias for compatibility
class PCamClassifier(PCamPredictor):
    pass