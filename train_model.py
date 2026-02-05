import os
import cv2
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration
DATA_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'Train')
TEST_CSV = os.path.join(DATA_DIR, 'Test.csv')
IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNELS = 3
NUM_CLASSES = 43
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

class TrafficSignDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.transform = transform
        self.labels = labels
        self.images = []
        print(f"Pre-loading {len(image_paths)} images into RAM...")
        for p in image_paths:
            try:
                img = Image.open(p).convert('RGB')
                self.images.append(img.copy())
                img.close()
            except Exception as e:
                print(f"Error loading {p}: {e}")
                self.images.append(Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT)))
        print("Pre-loading done.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def load_data_paths():
    print("Scanning dataset directories...")
    image_paths = []
    labels = []
    
    classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))], key=lambda x: int(x))
    for cls in classes:
        cls_path = os.path.join(TRAIN_DIR, cls)
        files = os.listdir(cls_path)
        for img_file in files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cls_path, img_file))
                labels.append(int(cls))
                
    return np.array(image_paths), np.array(labels)

def load_test_data_paths():
    print("Loading test CSV...")
    df = pd.read_csv(TEST_CSV)
    paths = df['Path'].apply(lambda x: os.path.join(DATA_DIR, x)).values
    labels = df['ClassId'].values
    return paths, labels

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2) # 32x32 -> 32x32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 32x32 -> 16x16
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 16x16 -> 16x16
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 16x16 -> 8x8
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # 8x8 -> 8x8
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2) # 8x8 -> 4x4
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, NUM_CLASSES)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_model():
    start_time = time.time()
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load Data
    full_paths, full_labels = load_data_paths()
    train_paths, val_paths, train_labels, val_labels = train_test_split(full_paths, full_labels, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")
    
    train_dataset = TrafficSignDataset(train_paths, train_labels, transform=transform)
    val_dataset = TrafficSignDataset(val_paths, val_labels, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows safety
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model Setup
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    print("Starting Training...")
    try:
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            epoch_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {epoch_acc:.2f}%")
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            print(f"Validation Acc: {val_acc:.2f}%")
            
        training_time = time.time() - start_time
        print(f"Training finished in {training_time:.2f}s")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model state...")
    except Exception as e:
        print(f"\nError during training: {e}")
    finally:
        # Save Model
        torch.save(model.state_dict(), 'traffic_sign_cnn.pth')
        print("Model saved to traffic_sign_cnn.pth")
    
    return model

def evaluate_on_test(model):
    print("\nEvaluating on Test Data...")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_paths, test_labels = load_test_data_paths()
    test_dataset = TrafficSignDataset(test_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {acc*100:.2f}%")

if __name__ == '__main__':
    model = train_model()
    evaluate_on_test(model)
