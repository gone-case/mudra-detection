import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")


def build_feature_extractor():
    """MobileNetV2 with classifier head removed — outputs 1280-dim feature vectors."""
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Keep only the features (conv layers + adaptive pool), drop the classifier
    feature_extractor = nn.Sequential(
        backbone.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    feature_extractor.eval()
    return feature_extractor.to(DEVICE)


class SVMPredictor:
    """
    Drop-in replacement for EnsemblePredictor.
    Uses a pretrained MobileNetV2 backbone to extract features and
    a trained sklearn SVM to classify Bharatanatyam mudras.
    """

    def __init__(self):
        self.class_names = []
        self.num_classes = 0
        self.svm = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._load_classes()
        self._load_feature_extractor()
        self._load_svm()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_classes(self):
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, "r") as f:
                self.class_names = json.load(f)
            self.num_classes = len(self.class_names)
            print(f"[SVMPredictor] Loaded {self.num_classes} classes: {self.class_names}")
        else:
            print("[SVMPredictor] Warning: class_names.json not found. Using defaults.")
            self.class_names = ["Pataka", "Tripataka", "Ardhapataka", "Kartarimukha"]
            self.num_classes = len(self.class_names)

    def _load_feature_extractor(self):
        print("[SVMPredictor] Building MobileNetV2 feature extractor …")
        self.feature_extractor = build_feature_extractor()
        print("[SVMPredictor] Feature extractor ready.")

    def _load_svm(self):
        if os.path.exists(SVM_MODEL_PATH):
            with open(SVM_MODEL_PATH, "rb") as f:
                self.svm = pickle.load(f)
            print(f"[SVMPredictor] SVM model loaded from {SVM_MODEL_PATH}")
        else:
            print(
                "[SVMPredictor] Warning: svm_model.pkl not found. "
                "Run training/train_svm.py first to train and save the SVM."
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _extract_features(self, frame_crop_bgr):
        """Convert BGR crop → normalised tensor → 1280-d feature vector."""
        import cv2
        img_rgb = cv2.cvtColor(frame_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = self.feature_extractor(tensor)

        return features.cpu().numpy()  # shape (1, 1280)

    def predict(self, frame_crop):
        """
        Accepts an OpenCV BGR frame crop.
        Returns: (top_class: str, top_confidence: float, top_3: list[dict])

        Mirrors the EnsemblePredictor.predict() interface for easy swap-in.
        """
        if frame_crop is None or frame_crop.size == 0:
            return None, 0.0, []

        if self.svm is None:
            print("[SVMPredictor] SVM not loaded — cannot predict.")
            return "Unknown", 0.0, []

        try:
            features = self._extract_features(frame_crop)  # (1, 1280)

            # Decision function gives raw distances; use them as a proxy for
            # confidence when probability=False. If model was trained with
            # probability=True, use predict_proba instead.
            if hasattr(self.svm, "predict_proba"):
                proba = self.svm.predict_proba(features)[0]  # shape (num_classes,)
            else:
                # Fallback: softmax over decision function scores
                scores = self.svm.decision_function(features)[0]
                proba = self._softmax(scores)

            # Sort by confidence descending
            sorted_idx = np.argsort(proba)[::-1]
            top_n = min(3, self.num_classes)

            top_prediction = self.class_names[sorted_idx[0]]
            top_confidence = float(proba[sorted_idx[0]])

            top_3 = [
                {
                    "mudra": self.class_names[sorted_idx[i]],
                    "confidence": float(proba[sorted_idx[i]])
                }
                for i in range(top_n)
            ]

            return top_prediction, top_confidence, top_3

        except Exception as e:
            print(f"[SVMPredictor] Prediction error: {e}")
            return "Unknown", 0.0, []

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()
