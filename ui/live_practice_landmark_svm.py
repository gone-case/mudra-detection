"""
live_practice_landmark_svm.py
-----------------------------
Live webcam practice page using strictly the SVM landmark model 
(highest accuracy without ensemble dilution).
"""

import streamlit as st
import cv2
import av
import threading
import time
import pickle
import json
import numpy as np

from pathlib import Path
from sqlalchemy import func

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from utils.hand_detector import HandDetector
from progress_tracker.tracker import ProgressTracker
from database.schema import SessionLocal, Mudra

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SVM_PATH   = MODELS_DIR / "landmark_svm.pkl"
CN_PATH    = MODELS_DIR / "class_names.json"

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_shared = {
    "prediction":  "Waiting...",
    "confidence":  0.0,
    "top3":        [],
    "meaning":     "",
    "tips":        "",
}
_lock = threading.Lock()

def landmarks_to_vector(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten()

# ---------------------------------------------------------------------------
# WebRTC video processor
# ---------------------------------------------------------------------------

class SVMLandmarkVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector  = HandDetector(max_num_hands=1)
        self.tracker   = ProgressTracker(user_id=1)
        self.db        = SessionLocal()
        
        # Load resources for raw SVM classification
        with open(CN_PATH, "r") as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)
        
        with open(SVM_PATH, "rb") as f:
            self.svm = pickle.load(f)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # 1. MediaPipe detection
        results = self.detector.process_frame(img)

        pred_class, conf, top_3 = None, 0.0, []
        
        # 2. Raw SVM classification
        if results and results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            vec = landmarks_to_vector(landmarks)
            vec_2d = vec.reshape(1, -1)
            
            # Predict
            probs = self.svm.predict_proba(vec_2d)[0]
            
            proba_sum = np.zeros(self.num_classes)
            for i, class_idx in enumerate(self.svm.classes_):
                if class_idx < self.num_classes:
                    proba_sum[class_idx] = probs[i]
                    
            pred_idx = np.argmax(proba_sum)
            conf = proba_sum[pred_idx]
            pred_class = self.class_names[pred_idx]
            
            # Sort top 3
            sorted_idx = np.argsort(proba_sum)[::-1]
            top_3 = [
                {
                    "mudra": self.class_names[sorted_idx[i]],
                    "confidence": float(proba_sum[sorted_idx[i]])
                }
                for i in range(min(3, self.num_classes))
            ]

        if pred_class and pred_class not in ("Unknown", None):
            # 3. DB lookup
            mudra_info = (
                self.db.query(Mudra)
                .filter(func.lower(Mudra.name) == pred_class.lower())
                .first()
            )
            meaning = mudra_info.meaning if mudra_info else ""
            tips    = mudra_info.improvement_tips if mudra_info else ""

            with _lock:
                _shared["prediction"] = pred_class
                _shared["confidence"] = conf
                _shared["top3"]       = top_3
                _shared["meaning"]    = meaning
                _shared["tips"]       = tips

            # 4. Progress tracking
            self.tracker.record_attempt(pred_class, conf, is_correct=True)

            # 5. Draw bounding box from hand crop
            _, bbox = self.detector.get_hand_crop(img, results)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 230, 118), 3)
                # Elegantly drawn label
                label = f"{pred_class}  {conf * 100:.1f}%"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
                cv2.rectangle(img, (x_min, max(0, y_min - text_h - 15)), 
                              (x_min + text_w + 10, max(0, y_min - text_h - 15) + text_h + 10), 
                              (0, 230, 118), -1)
                cv2.putText(img, label, (x_min + 5, max(0, y_min - text_h - 15) + text_h + 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

        # 6. Draw landmarks
        img = self.detector.draw_landmarks(img, results)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self):
        self.tracker.end_session()
        self.db.close()


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render_live_practice_svm():
    st.header("🔴 Live Practice — Standalone SVM")
    st.markdown(
        "Real-time mudra recognition optimized for speed and accuracy using **only the SVM model** "
        "trained on MediaPipe 21-keypoint hand landmarks. This bypasses the ensemble dilution effect."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        ctx = webrtc_streamer(
            key="mudra-svm-predictor",
            video_processor_factory=SVMLandmarkVideoProcessor,
        )

    with col2:
        st.subheader("Live Feedback")
        st.markdown("---")

        with _lock:
            pred    = _shared["prediction"]
            conf    = _shared["confidence"]
            meaning = _shared["meaning"]
            tips    = _shared["tips"]
            top3    = list(_shared["top3"])

        st.markdown("### Prediction")
        if pred == "Waiting...":
            st.info("⏳ Waiting for hand detection…")
        else:
            st.success(pred)

        st.markdown("### Confidence")
        st.progress(float(conf))

        st.markdown("### Meaning")
        st.write(meaning if meaning else "—")

        st.markdown("### Improvement Tips")
        st.write(tips if tips else "—")

        st.markdown("### Top 3 Predictions")
        if top3:
            for item in top3:
                st.write(f"{item['mudra']} — {item['confidence'] * 100:.1f}%")
        else:
            st.write("—")

    if ctx.state.playing:
        time.sleep(0.5)
        st.rerun()
