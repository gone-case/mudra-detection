"""
live_practice_landmark.py
-------------------------
Live webcam practice page using the 4-model landmark ensemble.
Uses recv() (not deprecated transform()) for streamlit-webrtc.
"""

import streamlit as st
import cv2
import av
import threading
import time

from sqlalchemy import func

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from utils.hand_detector import HandDetector
from ensemble.landmark_ensemble_predictor import LandmarkEnsemblePredictor
from progress_tracker.tracker import ProgressTracker
from database.schema import SessionLocal, Mudra


# ---------------------------------------------------------------------------
# Shared state — use a single dict so it is never reset by Streamlit reruns
# ---------------------------------------------------------------------------

_shared = {
    "prediction":  "Waiting...",
    "confidence":  0.0,
    "top3":        [],
    "meaning":     "",
    "tips":        "",
}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# WebRTC video processor
# ---------------------------------------------------------------------------

class LandmarkVideoProcessor(VideoProcessorBase):
    """
    Per-frame pipeline:
      Frame → HandDetector (draw landmarks) → LandmarkEnsemblePredictor
            → DB lookup → ProgressTracker
    """

    def __init__(self):
        self.detector  = HandDetector(max_num_hands=1)
        self.predictor = LandmarkEnsemblePredictor()
        self.tracker   = ProgressTracker(user_id=1)
        self.db        = SessionLocal()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # 1. MediaPipe detection — ONE call for both drawing AND prediction
        results = self.detector.process_frame(img)

        # 2. Ensemble prediction — reuse results, no second MediaPipe call
        pred_class, conf, top_3 = self.predictor.predict_from_landmarks(results)

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
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 230, 118), 2)
                label = f"{pred_class}  {conf * 100:.1f}%"
                cv2.putText(img, label, (x_min, max(y_min - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 118), 2)

        # 6. Draw landmarks
        img = self.detector.draw_landmarks(img, results)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self):
        self.tracker.end_session()
        self.db.close()


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render_live_practice_landmark():
    st.header("🔴 Live Practice — Landmark Ensemble")
    st.markdown(
        "Real-time mudra recognition using **4 classifiers** "
        "(SVM · Random Forest · KNN · Gradient Boosting) "
        "ensembled over MediaPipe 21-keypoint hand landmarks."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        ctx = webrtc_streamer(
            key="mudra-landmark-ensemble",
            video_processor_factory=LandmarkVideoProcessor,
        )

    with col2:
        st.subheader("Live Feedback")
        st.markdown("---")

        # Read snapshot of shared state under lock
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

    # Auto-refresh every 0.5 s while the stream is active
    if ctx.state.playing:
        time.sleep(0.5)
        st.rerun()
