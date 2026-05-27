"""
extract_to_csv.py
--------------------------
Extract MediaPipe hand landmarks from dataset images and save to CSV.
Handles unicode filenames gracefully using cv2.imdecode.
"""
import os
import sys
import json
import csv
import numpy as np
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
TASK_FILE  = ROOT / "hand_landmarker.task"
CSV_FILE   = DATA_DIR / "landmarks.csv"

def build_landmarker():
    base_opts = mp_python.BaseOptions(model_asset_path=str(TASK_FILE))
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(opts)

def landmarks_to_vector(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten()

def main():
    if not DATA_DIR.exists():
        print(f"[ERROR] Dataset directory missing: {DATA_DIR}")
        sys.exit(1)

    print("[Extract] Building MediaPipe HandLandmarker ...")
    landmarker = build_landmarker()

    class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    
    # Save the class names explicitly so we ensure it matches the CSV rows
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)
    print(f"[Extract] Found {len(class_names)} classes, saved to class_names.json")

    total_imgs = sum(len(list(d.glob("*"))) for d in class_dirs)
    print(f"[Extract] Starting extraction on approx {total_imgs} files...")

    header = ["label", "class_idx"] + [f"f_{i}" for i in range(63)]
    
    processed = 0
    skipped = 0
    success = 0

    with open(CSV_FILE, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for class_idx, class_dir in enumerate(class_dirs):
            label_name = class_dir.name
            img_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]

            for img_path in img_files:
                processed += 1
                if processed % 500 == 0:
                    print(f"  [{processed}/{total_imgs}] processing ...", flush=True)

                # Robust unicode file loading
                try:
                    stream = open(str(img_path), "rb")
                    bytes_arr = bytearray(stream.read())
                    numpyarray = np.asarray(bytes_arr, dtype=np.uint8)
                    bgr = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
                    stream.close()
                except Exception:
                    bgr = None

                if bgr is None:
                    skipped += 1
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:
                    result = landmarker.detect(mp_img)
                except Exception:
                    skipped += 1
                    continue

                if not result.hand_landmarks:
                    skipped += 1
                    continue

                vec = landmarks_to_vector(result.hand_landmarks[0])
                row = [label_name, class_idx] + vec.tolist()
                writer.writerow(row)
                success += 1

    landmarker.close()
    print(f"\n[Extract] Done!")
    print(f"  Total processed : {processed}")
    print(f"  Successful (hands found) : {success}")
    print(f"  Skipped (no hand/errors) : {skipped}")
    print(f"  Saved to : {CSV_FILE}")

if __name__ == "__main__":
    main()
