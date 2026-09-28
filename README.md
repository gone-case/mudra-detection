🖐️ MudraNet AI

Real-time Bharatanatyam mudra detection using Computer Vision and Machine Learning.

MudraNet AI recognizes 50 Bharatanatyam hand gestures from a live camera feed using MediaPipe hand landmarks, geometric feature extraction, and a custom machine-learning ensemble.

✨ Features

🎭 Recognizes 50 Bharatanatyam mudras

⚡ Real-time, low-latency detection

🖐️ MediaPipe-based hand landmark tracking

📐 Geometric feature extraction

🤖 Ensemble-based classification

📷 Webcam support

🔄 How It Works
Camera
  ↓
MediaPipe Hand Landmarks
  ↓
Feature Extraction
  ↓
ML Ensemble
  ↓
Mudra Prediction


The system extracts 21 hand landmarks, converts them into normalized geometric features, and uses the trained model to classify the detected mudra.

🛠️ Tech Stack

Python

OpenCV

MediaPipe

NumPy

Scikit-learn

🚀 Setup
git clone https://github.com/gone-case/mudra-detection.git
cd mudra-detection

python -m venv venv
# Activate the environment
pip install -r requirements.txt

python main.py

🎯 Applications

MudraNet AI can be used for Bharatanatyam learning, gesture-based interfaces, dance research, and digital preservation of traditional Indian art forms.

🔮 Future Scope

Dynamic gesture recognition

Two-hand detection

Confidence scores

Mobile/web deployment

Improved robustness across lighting and hand orientations

Built with ❤️ using Computer Vision, Machine Learning, and a passion for Bharatanatyam.
