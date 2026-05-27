"""
train_landmark_ensemble.py
--------------------------
Pipeline:
  1. Load features from dataset/landmarks.csv
  2. Train 4 classifiers on the landmark features:
       • SVM          (RBF kernel)
       • Random Forest
       • K-Nearest Neighbours
       • Gradient Boosting
  3. Soft-vote ensemble evaluated on a held-out val set.
  4. Save each model as a .pkl in models/.

Usage (from project root):
    python -m training.train_landmark_ensemble
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
CSV_FILE   = DATA_DIR / "landmarks.csv"
CN_FILE    = MODELS_DIR / "class_names.json"

MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_models():
    """Return dict of {name: unfitted_classifier}."""
    return {
        "svm": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=7, weights="distance", metric="euclidean", n_jobs=-1
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1,
            max_depth=4, random_state=42
        ),
    }

def soft_vote(proba_list):
    """Average probability matrices from multiple classifiers."""
    return np.mean(np.stack(proba_list, axis=0), axis=0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not CSV_FILE.exists():
        print(f"[ERROR] landmarks.csv not found. Please run: python -m training.extract_to_csv")
        sys.exit(1)

    if not CN_FILE.exists():
        print(f"[ERROR] class_names.json not found. Please run: python -m training.extract_to_csv")
        sys.exit(1)
        
    print("[train] Loading class names ...")
    with open(CN_FILE, "r") as f:
        class_names = json.load(f)

    # ---- 1. Load from CSV ----
    print(f"[train] Loading features from {CSV_FILE} ...")
    df = pd.read_csv(CSV_FILE)
    
    if df.empty:
        print("[ERROR] CSV is empty.")
        sys.exit(1)
        
    y = df['class_idx'].values
    feat_cols = [c for c in df.columns if c.startswith('f_')]
    X = df[feat_cols].values
    
    print(f"[train] Loaded {len(X)} samples with {X.shape[1]} features.")
    print(f"[train] Loaded {len(np.unique(y))} unique classes.")

    # ---- 2. Train / val split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[train] Train: {len(X_train)} | Val: {len(X_val)}")

    # ---- 3. Train each model ----
    models = make_models()
    trained = {}

    for name, clf in models.items():
        print(f"\n[train] Training {name} ...")
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        print(f"  {name} val accuracy: {acc * 100:.2f}%")
        trained[name] = clf

    # ---- 4. Soft-vote ensemble evaluation ----
    print("\n[train] Evaluating ensemble (soft voting) ...")
    proba_list = [clf.predict_proba(X_val) for clf in trained.values()]
    avg_proba  = soft_vote(proba_list)
    y_pred_ens = np.argmax(avg_proba, axis=1)
    ens_acc    = accuracy_score(y_val, y_pred_ens)
    print(f"  Ensemble val accuracy: {ens_acc * 100:.2f}%")
    
    # Generate report mapping actual unique classes found so there are no dimension mismatches
    unique_val_classes = np.unique(y_val)
    target_names = [class_names[i] for i in unique_val_classes]
    print("\n" + classification_report(y_val, y_pred_ens, target_names=target_names))

    # ---- 5. Save models ----
    for name, clf in trained.items():
        path = MODELS_DIR / f"landmark_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(clf, f)
        print(f"[train] Saved -> {path}")

    print("\n✅ All models saved. Dataset synced. Run app_landmark.py to start the app.")

if __name__ == "__main__":
    main()
