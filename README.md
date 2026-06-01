# 😊 Real-Time Emotion Detector

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.80+-red?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow&logoColor=white)
![Level](https://img.shields.io/badge/Level-Advanced-purple)

An advanced computer vision project that performs **real-time facial emotion recognition** from a live webcam feed using **DeepFace** and **OpenCV**. Detects 7 emotions with color-coded overlays, probability bars, and frame smoothing for stable output.

---

## Overview

Uses a pre-trained deep learning model via the DeepFace library to classify facial expressions in real time. Processes each frame from the webcam, smooths predictions over a 5-frame window, and overlays results directly on the video feed.

---

## Features

- Real-time webcam emotion detection at live FPS
- Detects **7 emotions**: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- Color-coded dominant emotion overlay:

  | Emotion | Color |
  |---------|-------|
  | Happy | 🟢 Green |
  | Sad | 🔵 Blue |
  | Angry | 🔴 Red |
  | Fear | 🟣 Purple |
  | Surprise | 🟡 Yellow |
  | Disgust | 🟠 Orange |
  | Neutral | ⚪ White |

- 5-frame rolling average for stable, flicker-free predictions
- Probability bars for all 7 emotions displayed on-screen
- Real-time FPS counter
- Press `S` to save a screenshot, `Q` to quit
- `analyze_image(path)` mode for static image analysis
- Automatically detects image path from command-line argument

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| DeepFace | Pre-trained emotion recognition model |
| OpenCV | Webcam capture, frame rendering, overlays |
| TensorFlow | Deep learning backend for DeepFace |
| NumPy | Frame processing and array operations |

---

## Project Structure

```
05_emotion_detector/
├── detect_emotions.py   # Main script (webcam + image modes)
├── requirements.txt     # Dependencies
└── screenshots/         # Saved screenshots (created on first save)
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Sherwin951/realtime-emotion-detector.git
cd realtime-emotion-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow and DeepFace are large packages (~500MB). Installation may take a few minutes.

### 3. Run webcam mode
```bash
python detect_emotions.py
```

### 4. Run on a static image
```bash
python detect_emotions.py path/to/image.jpg
```

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save a screenshot to `screenshots/` |

---

## How It Works

```
Webcam Frame
     │
     ▼
DeepFace.analyze()          ← Pre-trained CNN model (FER+ dataset)
     │
     ▼
Emotion Probabilities        ← 7 classes with confidence scores
     │
     ▼
5-Frame Smoothing            ← Rolling deque for stable output
     │
     ▼
OpenCV Overlay               ← Color-coded text + probability bars
     │
     ▼
Display + FPS Counter
```

---

## Requirements

- A working **webcam** (built-in or USB)
- Python 3.8 or higher
- ~2GB free disk space (TensorFlow + DeepFace models)
- For best results: good lighting and face clearly visible

---

## Performance

| Hardware | Approx. FPS |
|----------|-------------|
| MacBook M1/M2 | 15–25 FPS |
| Intel i5/i7 (CPU only) | 8–15 FPS |
| NVIDIA GPU (CUDA) | 25–40 FPS |

> DeepFace analyzes every 3rd frame to maintain smooth video on CPU.

---

## Troubleshooting

**Webcam not opening:**
```bash
# Try different camera index
# Edit detect_emotions.py: cv2.VideoCapture(0) → cv2.VideoCapture(1)
```

**TensorFlow installation error on Apple Silicon:**
```bash
pip install tensorflow-macos
pip install tensorflow-metal  # GPU acceleration
```

**DeepFace model download:**  
On first run, DeepFace automatically downloads the required model weights (~100MB). Ensure you have an internet connection.

---

## Author

**Sherwin Alle**  
M.S. Computer Science — California State University, Fresno  
[GitHub](https://github.com/Sherwin951) · [Email](mailto:alle.sherwin9999@gmail.com)
