import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modify the ResNet-50 model to accept 5-channel input
class vitEarlyFusion(nn.Module):
    def __init__(self, num_classes=2):
        super(vitEarlyFusion, self).__init__()
        self.model = timm.create_model('vit', pretrained=True)
        # Modify the first convolutional layer to accept 5 channels
        self.model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
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
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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
    model = vitEarlyFusion(num_classes=2).to(device)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001)

if __name__ == "__main__":
    main()

