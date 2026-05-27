import cv2
import json
import pickle
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Paths to existing models (so we don't duplicate files)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TASK_FILE  = ROOT / "hand_landmarker.task"
SVM_PATH   = MODELS_DIR / "landmark_svm.pkl"
CN_PATH    = MODELS_DIR / "class_names.json"

# ---------------------------------------------------------------------------
# Setup MediaPipe
# ---------------------------------------------------------------------------
def build_landmarker():
    base_opts = mp_python.BaseOptions(model_asset_path=str(TASK_FILE))
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def landmarks_to_vector(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten()

# ---------------------------------------------------------------------------
# Core UI Application Loop
# ---------------------------------------------------------------------------
def main():
    print("[Standalone Demo] Loading SVM model...")
    if not SVM_PATH.exists():
        print(f"[ERROR] SVM model safely not found at {SVM_PATH}")
        return
        
    with open(SVM_PATH, "rb") as f:
        svm_model = pickle.load(f)
        
    with open(CN_PATH, "r") as f:
        class_names = json.load(f)
        
    global_num_classes = len(class_names)
    print(f"[Standalone Demo] SVM ready, evaluating {global_num_classes} total classes.")
    
    print("[Standalone Demo] Starting MediaPipe HandLandmarker...")
    landmarker = build_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("---------------------------------------")
    print(" PRESS 'Q' OR ESC TO QUIT THE DEMO ")
    print("---------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for mirrored view
        frame = cv2.flip(frame, 1)

        # Convert to RGB (MediaPipe requirement)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        results = landmarker.detect(mp_img)

        # If a hand is detected
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            
            # --- 1. Extract & Classify ---
            vec = landmarks_to_vector(landmarks)
            vec_2d = vec.reshape(1, -1)
            
            # Use predict_proba for confidence bounding
            probs = svm_model.predict_proba(vec_2d)[0]
            
            # Robust extraction mapping for missing classes
            proba_sum = np.zeros(global_num_classes)
            for i, class_idx in enumerate(svm_model.classes_):
                if class_idx < global_num_classes:
                    proba_sum[class_idx] = probs[i]
            
            pred_idx = np.argmax(proba_sum)
            confidence = proba_sum[pred_idx]
            pred_class = class_names[pred_idx]

            # --- 2. Draw Landmark Skeleton ---
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

            # --- 3. Draw Beautiful Bounding Box & Label ---
            pad = 20
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (50, 205, 50), 3)

            # Elegant semi-transparent label background
            label_text = f"{pred_class} | {confidence*100:.1f}%"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
            
            label_y_start = max(0, y_min - text_h - 15)
            cv2.rectangle(
                frame, 
                (x_min, label_y_start), 
                (x_min + text_w + 10, label_y_start + text_h + 10), 
                (50, 205, 50), -1
            )
            cv2.putText(
                frame, label_text, 
                (x_min + 5, label_y_start + text_h + 3), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2
            )

        # Show Output
        cv2.imshow("MudraNet | High-Speed OpenCV Native Demo (SVM-only)", frame)

        # Break gracefully
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # Esc
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
