import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5):
        self.max_num_hands = max_num_hands

        # Load hand landmark model
        base_options = python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = self.detector.detect(mp_image)
        return results

    def get_hand_crop(self, frame, results, padding=20):
        if not results.hand_landmarks:
            return None, None

        h, w, _ = frame.shape
        landmarks = results.hand_landmarks[0]

        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        crop = frame[y_min:y_max, x_min:x_max]

        return crop, (x_min, y_min, x_max, y_max)

    def draw_landmarks(self, frame, results):
        if not results.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for landmarks in results.hand_landmarks:
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        return frame