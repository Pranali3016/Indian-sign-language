# 🤟 SignBridge AI - Real-Time Sign Language Recognition & Translation

An AI-powered web and desktop application that translates Indian/American Sign Language gestures into text and natural speech in real-time using **MediaPipe Holistic Landmark Tracking** and a **PyTorch Bidirectional GRU Neural Network**.

---

## ✨ Features
- **Ambidextrous & Distance Invariant**: Seamless recognition whether performed with the Left Hand or Right Hand.
- **Real-Time Skeleton Visualization**: Full tracking of Face Mesh, Upper Body Pose, and Hand Joints at 60 FPS.
- **Live Translation & Speech Synthesis**: Real-time sentence construction with natural Voice Read-Aloud (Text-to-Speech).
- **FastAPI + PyTorch Backend**: Low latency inference with confidence probability meters.
- **Glassmorphism Dark UI**: Responsive web interface with interactive gesture guide.

---

## 📂 Project Structure

```
sign-language-ai/
├── app.py                      # FastAPI Web Application Backend
├── features.py                 # Universal Invariant Feature Extraction Pipeline
├── train_robust_classifier.py  # SignNet Neural Network Trainer
├── record_data.py              # Gesture Dataset Recorder with Countdown HUD
├── optimized_realtime.py       # Desktop OpenCV Real-Time Visualizer
├── best_sign_model.pt          # Pretrained Neural Network Weights
├── requirements.txt            # Python Dependencies
├── Dockerfile                  # Container Deployment Config
├── templates/
│   └── index.html              # Responsive Web Dashboard
└── static/
    ├── app.js                  # MediaPipe Client Engine & Speech Logic
    ├── style.css               # Modern Glassmorphic Stylesheet
    └── model_metadata.json     # Model Configuration & Metadata
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```
👉 Open your browser at: **`http://localhost:8050`**

### 3. Run the Desktop OpenCV Window
```bash
python optimized_realtime.py
```

---

## 🧠 Model Architecture & Invariant Features
- **Feature Vector (162 Dimensions)**:
  - Primary Hand Shape (Wrist-Centered & Palm-Scaled): 63
  - Primary Hand Elevation relative to Nose: 3
  - Secondary Hand Shape: 63
  - Secondary Hand Elevation relative to Nose: 3
  - Upper Body Posture Anchors: 30
- **Network (`SignNet`)**:
  - 2-Layer Bidirectional GRU (Hidden Dim: 64, Dropout: 0.2)
  - Linear(128 -> 64) -> LayerNorm -> ReLU -> Dropout(0.25) -> Linear(64 -> 8)

---

## 📜 Supported Gestures
- 🧍 **Alone**
- 📞 **Call**
- 🌸 **Flower**
- 🍲 **Food**
- 👍 **I am good**
- 👌 **Ok Fine**
- ✋ **Stop**
- ⚠️ **There is Gun**
