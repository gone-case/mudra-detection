"""
train_svm.py
------------
Trains an SVM classifier on top of MobileNetV2 features
extracted from the mudra dataset.

Usage (from project root):
    python -m training.train_svm
    -- or --
    python training/train_svm.py
"""

import os
import sys
import json
import pickle
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "dataset")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SVM_SAVE_PATH        = os.path.join(MODELS_DIR, "svm_model.pkl")
CLASS_NAMES_SAVE_PATH = os.path.join(MODELS_DIR, "class_names.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Feature extractor — MobileNetV2 backbone (no classifier head)
# ---------------------------------------------------------------------------

def build_feature_extractor():
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    extractor = nn.Sequential(
        backbone.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    extractor.eval()
    return extractor.to(DEVICE)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_data_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def extract_features(loader, extractor):
    """Run all images through the extractor and collect feature vectors + labels."""
    all_features = []
    all_labels   = []

    total_batches = len(loader)
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            feats = extractor(imgs)           # (B, 1280)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                print(f"  Extracting features … batch {i+1}/{total_batches}", flush=True)

    return np.vstack(all_features), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Sanity check ----
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print(f"[train_svm] Dataset directory '{DATA_DIR}' is empty or missing.")
        print("Please add images organised as: dataset/<class_name>/<image_files>")
        sys.exit(1)

    # ---- Load dataset ----
    transform   = get_data_transform()
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)
    print(f"[train_svm] Found {len(full_dataset)} images across {num_classes} classes: {class_names}")

    # ---- Train / val split  ----
    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Build feature extractor ----
    print("[train_svm] Building MobileNetV2 feature extractor …")
    extractor = build_feature_extractor()

    # ---- Extract features ----
    print("[train_svm] Extracting training features …")
    X_train, y_train = extract_features(train_loader, extractor)

    print("[train_svm] Extracting validation features …")
    X_val, y_val = extract_features(val_loader, extractor)

    print(f"[train_svm] Feature shapes — train: {X_train.shape}, val: {X_val.shape}")

    # ---- Train SVM ----
    print("\n[train_svm] Training SVM (kernel=rbf, C=10, probability=True) …")
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,   # enables predict_proba for confidence scores
        class_weight="balanced",
        verbose=True,
    )
    svm.fit(X_train, y_train)
    print("[train_svm] SVM training complete.")

    # ---- Evaluate ----
    y_pred = svm.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)
    print(f"\n[train_svm] Validation Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_val, y_pred, target_names=class_names))

    # ---- Save SVM ----
    with open(SVM_SAVE_PATH, "wb") as f:
        pickle.dump(svm, f)
    print(f"[train_svm] SVM saved → {SVM_SAVE_PATH}")

    # ---- Save class names ----
    with open(CLASS_NAMES_SAVE_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"[train_svm] Class names saved → {CLASS_NAMES_SAVE_PATH}")


if __name__ == "__main__":
    main()
