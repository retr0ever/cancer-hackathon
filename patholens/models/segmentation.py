import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage import feature, morphology, measure  # FIXED: measurement -> measure

# --- 1. The Deep Learning Model (UNet) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """ Standard UNet Architecture commonly used with NuInsSeg/MoNuSeg """
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.up1 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up3 = DoubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(self.maxpool(x1))
        x3 = self.down2(self.maxpool(x2))
        x4 = self.down3(self.maxpool(x3))
        
        x = self.upsample(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x)
        logits = self.outc(x)
        return torch.sigmoid(logits)

# --- 2. The Segmenter Class ---
class NucleiSegmenter:
    def __init__(self):
        # In a real app, you would load weights here:
        # self.model = UNet()
        # self.model.load_state_dict(torch.load("nuinsseg_weights.pth"))
        self.model = None 

    def segment(self, image):
        """
        Hybrid Approach: Uses Deep Learning if available, falls back to 
        robust Morphological processing.
        """
        img_np = np.array(image)
        
        # --- PLAN A: Deep Learning (Placeholder) ---
        if self.model is not None:
            # Run UNet inference here
            pass

        # --- PLAN B: Enhanced Color Deconvolution (Robust Fallback) ---
        # 1. Hematoxylin channel separation (nuclei are blue-ish)
        # Simple approximation without full stain unmixing
        red = img_np[:,:,0].astype(float)
        blue = img_np[:,:,2].astype(float)
        
        # Nuclei have high Blue / Low Red ratio
        nuclei_map = blue - red
        
        # 2. Thresholding (Otsu)
        nuclei_map = np.clip(nuclei_map, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(nuclei_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Morphological cleanup (remove noise)
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=20)
        binary = morphology.binary_opening(binary, morphology.disk(2))
        
        # 4. Watershed to split touching cells
        distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
        _, local_max = cv2.threshold(distance, 0.4 * distance.max(), 255, 0)
        local_max = np.uint8(local_max)
        
        # Connected components
        num_labels, labels = cv2.connectedComponents(local_max)
        
        # Watershed
        markers = cv2.watershed(img_np, labels.astype(np.int32))
        
        # Extract individual nuclei
        nuclei_masks = []
        clean_labels = np.zeros_like(labels)
        
        for i in range(1, num_labels):
            mask = (markers == i).astype(np.uint8) * 255
            if np.sum(mask) > 0:
                nuclei_masks.append(mask)
                clean_labels[markers == i] = i

        return nuclei_masks, clean_labels, None

def compute_nuclei_statistics(nuclei_list):
    """Compute stats for the grading logic."""
    if not nuclei_list:
        return {'count': 0, 'mean_area': 0, 'std_area': 0}
        
    areas = [np.sum(n > 0) for n in nuclei_list]
    return {
        'count': len(nuclei_list),
        'mean_area': np.mean(areas),
        'std_area': np.std(areas)
    }

# Function alias to support older imports if necessary
def segment_nuclei(image):
    seg = NucleiSegmenter()
    return seg.segment(image)