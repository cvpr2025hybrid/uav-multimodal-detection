import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define separate networks for RGB, thermal, and event channels
class RGBNet(nn.Module):
    def __init__(self, num_classes=2):
        super(RGBNet, self).__init__()
        self.model = timm.create_model('vit', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ThermalNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ThermalNet, self).__init__()
        self.model = timm.create_model('vit', pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class EventNet(nn.Module):
    def __init__(self, num_classes=2):
        super(EventNet, self).__init__()
        self.model = timm.create_model('vit', pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Late fusion model with soft voting
class LateFusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(LateFusionModel, self).__init__()
        self.rgb_net = RGBNet(num_classes)
        self.thermal_net = ThermalNet(num_classes)
        self.event_net = EventNet(num_classes)

    def forward(self, rgb, thermal, event):
        # Get predictions from all three models
        rgb_output = self.rgb_net(rgb)
        thermal_output = self.thermal_net(thermal)
        event_output = self.event_net(event)
        
        # Apply softmax to get probabilities
        rgb_probs = F.softmax(rgb_output, dim=1)
        thermal_probs = F.softmax(thermal_output, dim=1)
        event_probs = F.softmax(event_output, dim=1)
        
        # Soft voting: average the probabilities
        combined_probs = (rgb_probs + thermal_probs + event_probs) / 3
        return combined_probs

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for rgb_images, thermal_images, event_images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            rgb_images, thermal_images, event_images, labels = rgb_images.to(device), thermal_images.to(device), event_images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(rgb_images, thermal_images, event_images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
        
        train_acc = train_correct / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Validate the model
        val_acc = evaluate_model(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        print(f"Validation Accuracy: {val_acc:.4f}, Best Validation Accuracy: {best_val_acc:.4f}")

# Validation function
def evaluate_model(model, val_loader):
    model.eval()
    val_correct = 0

    with torch.no_grad():
        for rgb_images, thermal_images, event_images, labels in val_loader:
            rgb_images, thermal_images, event_images, labels = rgb_images.to(device), thermal_images.to(device), event_images.to(device), labels.to(device)
            outputs = model(rgb_images, thermal_images, event_images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    return val_acc

# Main function
def main():
    # Define transforms
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    transforms.ToTensor()
    ])

    # Load dataset
    train_dataset = 5ChannelDataset(root_dir='./train', transform=transform)
    val_dataset = 5ChannelDataset(root_dir='./val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the model
    model = LateFusionModel(num_classes=2).to(device)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)

if __name__ == "__main__":
    main()