import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hybrid Fusion Model using timm ResNet-50
class HybridFusionResNet(nn.Module):
    def __init__(self, fusion_layer, num_classes=2):
        super(HybridFusionResNet, self).__init__()
        self.fusion_layer = fusion_layer

        # Initialize separate branches for each modality using timm
        self.rgb_branch = timm.create_model('resnet50', pretrained=True, features_only=True)
        self.thermal_branch = timm.create_model('resnet50', pretrained=True, features_only=True)
        self.event_branch = timm.create_model('resnet50', pretrained=True, features_only=True)

        # Modify the input layer for thermal and event branches (1 channel input)
        self.thermal_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.event_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Define fusion layer (1x1 convolution after concatenation)
        self.fusion_conv = nn.Conv2d(2048 * 3, 2048, kernel_size=1, stride=1, padding=0)

        # Shared layers after fusion
        self.shared_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, rgb, thermal, event):
        # Extract features up to the selected fusion layer
        rgb_features = self.rgb_branch(rgb)[self.fusion_layer]
        thermal_features = self.thermal_branch(thermal)[self.fusion_layer]
        event_features = self.event_branch(event)[self.fusion_layer]

        # Concatenate features from all branches
        fused_features = torch.cat((rgb_features, thermal_features, event_features), dim=1)
        fused_features = self.fusion_conv(fused_features)

        # Shared processing after fusion
        shared_features = self.shared_layers(fused_features)
        pooled_features = shared_features.view(shared_features.size(0), -1)
        output = self.fc(pooled_features)

        return output

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for rgb, thermal, event, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            rgb, thermal, event, labels = rgb.to(device), thermal.to(device), event.to(device), labels.to(device)
            
            outputs = model(rgb, thermal, event)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
        
        train_acc = train_correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_acc = evaluate_model(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_layer_{model.fusion_layer}.pth')
        print(f"Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")

def evaluate_model(model, val_loader):
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for rgb, thermal, event, labels in val_loader:
            rgb, thermal, event, labels = rgb.to(device), thermal.to(device), event.to(device), labels.to(device)
            outputs = model(rgb, thermal, event)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
    return val_correct / len(val_loader.dataset)

def main():
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    transforms.ToTensor()
    ])
    train_dataset = 5ChannelDataset('./train', transform=transform)
    val_dataset = 5ChannelDataset('./val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Experiment with different fusion layers
    fusion_layers = [2, 3, 4]  # Layers to experiment with in `features_only`
    for layer in fusion_layers:
        print(f"\nTraining with fusion at layer {layer}")
        model = HybridFusionResNet(fusion_layer=layer, num_classes=2).to(device)
        train_model(model, train_loader, val_loader, num_epochs=50)

if __name__ == "__main__":
    main()
