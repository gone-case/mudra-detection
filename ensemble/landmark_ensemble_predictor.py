"""
landmark_ensemble_predictor.py
-------------------------------
Loads 4 sklearn classifiers trained on MediaPipe hand-landmark features
and combines their predictions via soft voting.

Drop-in interface (same as EnsemblePredictor / SVMPredictor):
    predictor = LandmarkEnsemblePredictor()
    pred_class, confidence, top_3 = predictor.predict(bgr_frame_crop)

NOTE: predict() here accepts the FULL frame (not just a crop) because
landmark detection is done internally — no need to crop first.
"""

import os
import json
import pickle
from pathlib import Path

import cv2
# HistGradientBoostingClassifier is used by train_landmark_ensemble.py (fast GBM)
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TASK_FILE  = ROOT / "hand_landmarker.task"

MODEL_NAMES = ["svm", "random_forest", "knn", "gradient_boosting"]


# ---------------------------------------------------------------------------
# Landmark normalisation (must match train_landmark_ensemble.py)
# ---------------------------------------------------------------------------

def landmarks_to_vector(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]                           # translate to wrist origin
    scale = np.linalg.norm(pts[9]) + 1e-6  # scale by wrist→middle-MCP distance
    pts /= scale
    return pts.flatten()                    # (63,)


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class LandmarkEnsemblePredictor:
    """
    4-model soft-voting ensemble that operates on MediaPipe hand landmarks.
    Much faster and typically more accurate than image-based approaches.
    """

    def __init__(self):
        self.class_names: list[str] = []
        self.num_classes: int       = 0
        self.models: dict           = {}

        self._load_classes()
        self._load_landmarker()
        self._load_models()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_classes(self):
        path = MODELS_DIR / "class_names.json"
        if path.exists():
            with open(path) as f:
                self.class_names = json.load(f)
            self.num_classes = len(self.class_names)
            print(f"[LandmarkEnsemble] {self.num_classes} classes: {self.class_names}")
        else:
            print("[LandmarkEnsemble] ⚠ class_names.json not found — using defaults.")
            self.class_names = ["Pataka", "Tripataka", "Ardhapataka", "Kartarimukha"]
            self.num_classes = len(self.class_names)

    def _load_landmarker(self):
        """Only used as fallback when predict(frame) is called directly."""
        if not TASK_FILE.exists():
            raise FileNotFoundError(f"hand_landmarker.task not found at {TASK_FILE}")

        base_opts = mp_python.BaseOptions(model_asset_path=str(TASK_FILE))
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        print("[LandmarkEnsemble] MediaPipe HandLandmarker ready (fallback mode).")

    def _load_models(self):
        loaded = []
        for name in MODEL_NAMES:
            path = MODELS_DIR / f"landmark_{name}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self.models[name] = pickle.load(f)
                loaded.append(name)
            else:
                print(f"[LandmarkEnsemble] ⚠ {path.name} not found — skipping.")

        if not self.models:
            print(
                "[LandmarkEnsemble] ⚠ No models loaded. "
                "Run training/train_landmark_ensemble.py first."
            )
        else:
            print(f"[LandmarkEnsemble] Loaded models: {loaded}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _get_landmark_vector(self, bgr_frame):
        """Run MediaPipe on a BGR frame and return a 63-d feature vector, or None."""
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self.landmarker.detect(mp_img)
        except Exception as e:
            print(f"[LandmarkEnsemble] Detection error: {e}")
            return None

        if not result.hand_landmarks:
            return None

        return landmarks_to_vector(result.hand_landmarks[0])

    def _classify(self, vec: np.ndarray):
        """Core classification from a 63-d normalised landmark vector."""
        if not self.models:
            return "Unknown", 0.0, []

        vec_2d = vec.reshape(1, -1)  # (1, 63)

        proba_sum = np.zeros(self.num_classes)
        for clf in self.models.values():
            probs = clf.predict_proba(vec_2d)[0]
            # Map model-specific classes to global class list
            for i, class_idx in enumerate(clf.classes_):
                if class_idx < self.num_classes:
                    proba_sum[class_idx] += probs[i]

        avg_proba  = proba_sum / len(self.models)
        sorted_idx = np.argsort(avg_proba)[::-1]
        top_n      = min(3, self.num_classes)

        top_class      = self.class_names[sorted_idx[0]]
        top_confidence = float(avg_proba[sorted_idx[0]])
        top_3 = [
            {
                "mudra":      self.class_names[sorted_idx[i]],
                "confidence": float(avg_proba[sorted_idx[i]])
            }
            for i in range(top_n)
        ]
        return top_class, top_confidence, top_3

    def predict_from_landmarks(self, mediapipe_results):
        """
        Preferred method — accepts MediaPipe HandLandmarker results already
        obtained by HandDetector.process_frame(). Avoids running MediaPipe twice.

        Parameters
        ----------
        mediapipe_results : mediapipe HandLandmarkerResult
            The object returned by HandDetector.process_frame().

        Returns
        -------
        (top_class: str, top_confidence: float, top_3: list[dict])
        """
        if not mediapipe_results or not mediapipe_results.hand_landmarks:
            return None, 0.0, []

        try:
            vec = landmarks_to_vector(mediapipe_results.hand_landmarks[0])
            return self._classify(vec)
        except Exception as e:
            print(f"[LandmarkEnsemble] predict_from_landmarks error: {e}")
            return "Unknown", 0.0, []

    def predict(self, frame):
        """
        Fallback — runs MediaPipe internally on a BGR frame.
        Prefer predict_from_landmarks() to avoid double MediaPipe inference.
        """
        if frame is None or frame.size == 0:
            return None, 0.0, []

        if not self.models:
            return "Unknown", 0.0, []

        try:
            vec = self._get_landmark_vector(frame)
            if vec is None:
                return None, 0.0, []
            return self._classify(vec)
        except Exception as e:
            print(f"[LandmarkEnsemble] Predict error: {e}")
            return "Unknown", 0.0, []
