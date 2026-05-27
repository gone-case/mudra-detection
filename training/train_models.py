import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Define Custom CNN
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 256) # Assuming 224x224 input
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loaders(batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load ALL images from dataset folder
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    print(f"Total images loaded: {len(full_dataset)}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = full_dataset.classes
    return train_loader, val_loader, class_names

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, model_name="model"):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
    # Save the model
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}.pth"))
    print(f"Saved {model_name}.pth")
    return model

def main():
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print(f"Dataset directory '{DATA_DIR}' is empty. Please add training data organized by class folder.")
        return
        
    train_loader, val_loader, class_names = get_data_loaders()
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Set specific epochs for different architectures
    CNN_EPOCHS = 6
    TRANSFER_EPOCHS = 3
    
    # 1. Custom CNN
    print("\n--- Training Custom CNN ---")
    custom_cnn = CustomCNN(num_classes).to(DEVICE)
    train_model(custom_cnn, train_loader, val_loader, criterion, optim.Adam(custom_cnn.parameters(), lr=0.001), num_epochs=CNN_EPOCHS, model_name="custom_cnn_1")
    
    # 2. MobileNetV2
    print("\n--- Training MobileNetV2 ---")
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)
    mobilenet = mobilenet.to(DEVICE)
    train_model(mobilenet, train_loader, val_loader, criterion, optim.Adam(mobilenet.parameters(), lr=0.0001), num_epochs=TRANSFER_EPOCHS, model_name="mobilenet_v2_1")
    
    # 3. ResNet50
    print("\n--- Training ResNet50 ---")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(DEVICE)
    train_model(resnet, train_loader, val_loader, criterion, optim.Adam(resnet.parameters(), lr=0.0001), num_epochs=TRANSFER_EPOCHS, model_name="resnet50_1")
    
    # 4. EfficientNet
    print("\n--- Training EfficientNet ---")
    efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
    efficientnet = efficientnet.to(DEVICE)
    train_model(efficientnet, train_loader, val_loader, criterion, optim.Adam(efficientnet.parameters(), lr=0.0001), num_epochs=TRANSFER_EPOCHS, model_name="efficientnet_1")
    
    # Save classes map
    import json
    with open(os.path.join(MODELS_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

if __name__ == "__main__":
    main()
