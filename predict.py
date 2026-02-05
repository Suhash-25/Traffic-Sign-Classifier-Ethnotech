import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import random
import sys

# Constants
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CLASSES = 43
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GTSRB Class Names
CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(128)
        
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

def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            
        class_id = predicted.item()
        class_name = CLASSES.get(class_id, "Unknown")
        return class_id, class_name
    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return None, None

def main():
    model = SimpleCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load('traffic_sign_cnn.pth', map_location=DEVICE))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if len(sys.argv) > 1:
        # Predict supplied image
        img_path = sys.argv[1]
        print(f"Predicting: {img_path}")
        cid, cname = predict_image(img_path, model)
        print(f"Result: Class {cid} - {cname}")
    else:
        # Pick random test image
        print("No image provided. Picking a random image from Test folder...")
        dataset_dir = 'dataset'
        test_dir = os.path.join(dataset_dir, 'Test')
        
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg'))]
            if images:
                rand_img = random.choice(images)
                full_path = os.path.join(test_dir, rand_img)
                print(f"Selected Image: {rand_img}")
                cid, cname = predict_image(full_path, model)
                print(f"Prediction: Class {cid} -> {cname}")
                
                # Try to verify against Test.csv if possible
                try:
                    import pandas as pd
                    test_csv = os.path.join(dataset_dir, 'Test.csv')
                    df = pd.read_csv(test_csv)
                    # Test.csv paths are like 'Test/00000.png'
                    csv_path = f"Test/{rand_img}"
                    row = df[df['Path'] == csv_path]
                    if not row.empty:
                        true_id = row.iloc[0]['ClassId']
                        true_name = CLASSES.get(true_id, "Unknown")
                        print(f"True Label: Class {true_id} -> {true_name}")
                        if cid == true_id:
                            print("✅ Correct Prediction!")
                        else:
                            print("❌ Incorrect Prediction.")
                except:
                    pass
            else:
                print("No images found in dataset/Test")
        else:
            print("dataset/Test directory not found.")

if __name__ == '__main__':
    main()
