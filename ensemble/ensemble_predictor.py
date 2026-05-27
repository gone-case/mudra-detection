import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

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
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
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


class EnsemblePredictor:
    def __init__(self):
        self.class_names = []
        self.num_classes = 0
        self.models = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.load_classes()
        self.init_models()
        self.softmax = nn.Softmax(dim=1)

    def load_classes(self):
        class_file = os.path.join(MODELS_DIR, "class_names.json")
        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                self.class_names = json.load(f)
            self.num_classes = len(self.class_names)
        else:
            print("Warning: class_names.json not found in models/ directory. Using defaults.")
            self.class_names = ["Pataka", "Tripataka", "Ardhapataka", "Kartarimukha"]
            self.num_classes = 4

    def init_models(self):
        print(f"Loading Ensemble Models for {self.num_classes} classes...")
        try:
            # 1. Custom CNN
            cnn = CustomCNN(self.num_classes).to(DEVICE)
            cnn_path = os.path.join(MODELS_DIR, "custom_cnn_1.pth")
            if os.path.exists(cnn_path):
                cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
            cnn.eval()
            self.models['custom_cnn'] = cnn

            # 2. MobileNetV2
            mob = models.mobilenet_v2()
            mob.classifier[1] = nn.Linear(mob.last_channel, self.num_classes)
            mob = mob.to(DEVICE)
            mob_path = os.path.join(MODELS_DIR, "mobilenet_v2_1.pth")
            if os.path.exists(mob_path):
                mob.load_state_dict(torch.load(mob_path, map_location=DEVICE))
            mob.eval()
            self.models['mobilenet_v2'] = mob

            # 3. ResNet50
            res = models.resnet50()
            res.fc = nn.Linear(res.fc.in_features, self.num_classes)
            res = res.to(DEVICE)
            res_path = os.path.join(MODELS_DIR, "resnet50_1.pth")
            if os.path.exists(res_path):
                res.load_state_dict(torch.load(res_path, map_location=DEVICE))
            res.eval()
            self.models['resnet50'] = res

            # 4. EfficientNet
            eff = models.efficientnet_b0()
            eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, self.num_classes)
            eff = eff.to(DEVICE)
            eff_path = os.path.join(MODELS_DIR, "efficientnet_1.pth")
            if os.path.exists(eff_path):
                eff.load_state_dict(torch.load(eff_path, map_location=DEVICE))
            eff.eval()
            self.models['efficientnet'] = eff
            
            print("Ensemble Models Loaded Successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")

    def predict(self, frame_crop):
        """
        Takes an OpenCV BGR frame crop, applies transform, predicts using all models (Soft Voting).
        """
        if frame_crop is None or frame_crop.size == 0:
            return None, 0.0, []

        try:
            # Convert OpenCV frame to PIL Image
            import cv2
            img_rgb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Preprocess
            input_tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)
            
            # Predict
            probs_sum = torch.zeros(1, self.num_classes).to(DEVICE)
            
            with torch.no_grad():
                for name, model in self.models.items():
                    out = model(input_tensor)
                    probs = self.softmax(out)
                    probs_sum += probs
                    
            # Soft voting (average)
            avg_probs = probs_sum / len(self.models)
            
            # Extract top 3
            topk_vals, topk_idx = torch.topk(avg_probs, min(3, self.num_classes), dim=1)
            
            topk_vals = topk_vals.cpu().numpy()[0]
            topk_idx = topk_idx.cpu().numpy()[0]
            
            top_prediction = self.class_names[topk_idx[0]]
            top_confidence = float(topk_vals[0])
            
            top_3 = []
            for i in range(len(topk_idx)):
                top_3.append({
                    "mudra": self.class_names[topk_idx[i]],
                    "confidence": float(topk_vals[i])
                })
                
            return top_prediction, top_confidence, top_3
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown", 0.0, []
