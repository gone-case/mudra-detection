import streamlit as st
import cv2
import av
import threading
import time
from sqlalchemy import func

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from utils.hand_detector import HandDetector
from ensemble.ensemble_predictor import EnsemblePredictor
from progress_tracker.tracker import ProgressTracker
from database.schema import SessionLocal, Mudra


# Global references for shared state
lock = threading.Lock()
current_prediction = "Waiting..."
current_confidence = 0.0
top_3_predictions = []
current_meaning = ""
current_tips = ""


class MudraVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(max_num_hands=1)
        self.predictor = EnsemblePredictor()
        self.tracker = ProgressTracker(user_id=1)
        self.db = SessionLocal()

    def transform(self, frame):
        global current_prediction, current_confidence
        global top_3_predictions, current_meaning, current_tips

        img = frame.to_ndarray(format="bgr24")

        # 1. Hand Detection
        results = self.detector.process_frame(img)
        crop, bbox = self.detector.get_hand_crop(img, results)

        if crop is not None:
            # 2. Prediction
            pred_class, conf, top_3 = self.predictor.predict(crop)

            with lock:
                current_prediction = pred_class
                current_confidence = conf
                top_3_predictions = top_3

            # 3. Get Meaning from DB (case-insensitive)
            mudra_info = (
                self.db.query(Mudra)
                .filter(func.lower(Mudra.name) == pred_class.lower())
                .first()
            )

            if mudra_info:
                with lock:
                    current_meaning = mudra_info.meaning
                    current_tips = mudra_info.improvement_tips

            # 4. Track progress
            is_correct = True
            self.tracker.record_attempt(pred_class, conf, is_correct)

            # Draw Bbox
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cv2.putText(
                img,
                f"{pred_class} {conf*100:.1f}%",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # Draw landmarks
            img = self.detector.draw_landmarks(img, results)

        return img

    def on_ended(self):
        self.tracker.end_session()
        self.db.close()


def render_live_practice():
    st.header("🔴 Live Practice Area")
    st.markdown("Turn on your webcam and show a mudra to the camera.")

    col1, col2 = st.columns([2, 1])

    with col1:
        webrtc_streamer(
            key="mudra-recognition",
            video_transformer_factory=MudraVideoTransformer,
            async_transform=True,
        )

    with col2:
        st.subheader("Live Feedback")
        st.markdown("---")

        placeholder = st.empty()

        while True:
            with lock:
                pred = current_prediction
                conf = current_confidence
                meaning = current_meaning
                tips = current_tips
                top3 = top_3_predictions

            with placeholder.container():

                st.markdown("### Prediction")
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
                        st.write(
                            f"{item['mudra']} — {item['confidence']*100:.1f}%"
                        )
                else:
                    st.write("—")

            time.sleep(0.2)