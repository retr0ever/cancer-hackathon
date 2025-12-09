import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
import medmnist
from medmnist import INFO, PathMNIST
import os

def train_model():
    print("--- PathoLens Training Pipeline ---")
    
    # 1. Configuration
    BATCH_SIZE = 128
    LR = 0.001
    EPOCHS = 3  # Increase to 10 for better results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data (PathMNIST = Colon Pathology)
    # This automatically downloads the dataset (~300MB)
    info = INFO['pathmnist']
    DataClass = PathMNIST

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    print("Downloading/Loading Data...")
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    val_dataset = DataClass(split='val', transform=data_transform, download=True)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE*2, shuffle=False)

    # 3. Setup Model (ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify the first layer to handle the specific normalization of medical images if needed,
    # but standard ResNet is fine. We MUST modify the last layer.
    num_ftrs = model.fc.in_features
    # PathMNIST has 9 classes, but we can treat non-tumour (labels 0-7) vs tumour (label 8)
    # Or, to stick to your binary pipeline, we train on the 9 classes and map them later.
    # For simplicity in this hackathon context, let's train a direct binary mapper.
    # NOTE: PathMNIST is effectively PCam. 
    model.fc = nn.Linear(num_ftrs, 9) 
    model.to(device)

    # 4. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze().long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(train_loader), 'acc': 100.*correct/total})
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze().long()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f" Validation Accuracy: {val_acc:.2f}%")
        scheduler.step()

    # 5. Save the 'Brain'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "models", "patholens_model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {os.path.relpath(save_path)}")

if __name__ == '__main__':
    train_model()