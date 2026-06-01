# 😊 Real-Time Emotion Detector

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv&logoColor=white)
![Level](https://img.shields.io/badge/Level-Advanced-purple)

An advanced computer vision project that performs **real-time facial emotion recognition** from a live webcam feed using a **HuggingFace Vision Transformer** and **OpenCV**. Detects 7 emotions with color-coded overlays, probability bars, and frame smoothing for stable output.

---

## Overview

Uses the pre-trained `trpakov/vit-face-expression` ViT model via HuggingFace Transformers (PyTorch backend) to classify facial expressions in real time. OpenCV Haar Cascade handles face detection. Predictions are smoothed over a 5-frame window and overlaid directly on the video feed.

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
| HuggingFace Transformers | Pre-trained ViT emotion model |
| PyTorch | Deep learning backend |
| OpenCV | Webcam capture, face detection, overlays |
| Pillow | Image format conversion |
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

> **Note:** The HuggingFace model (`trpakov/vit-face-expression`) is ~343MB and downloads automatically on first run.

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
| `S` | Save a screenshot |

---

## How It Works

```
Webcam Frame
     │
     ▼
OpenCV Haar Cascade         ← Face detection
     │
     ▼
ViT (trpakov/vit-face-expression)  ← HuggingFace image-classification pipeline
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
- ~500MB free disk space (PyTorch + ViT model weights)
- For best results: good lighting and face clearly visible

> **macOS users:** Grant Terminal camera access in **System Settings → Privacy & Security → Camera**

---

## Performance

| Hardware | Approx. FPS |
|----------|-------------|
| MacBook M1/M2 | 15–25 FPS |
| Intel i5/i7 (CPU only) | 8–15 FPS |
| NVIDIA GPU (CUDA) | 25–40 FPS |

> Emotion analysis runs every 5th frame to maintain smooth video on CPU.

---

## Troubleshooting

**Webcam not opening:**
```bash
# Try different camera index
# Edit detect_emotions.py: cv2.VideoCapture(0) → cv2.VideoCapture(1)
```

**macOS camera permission denied:**
```
System Settings → Privacy & Security → Camera → enable Terminal
```

**First run slow:**  
The ViT model (~343MB) downloads from HuggingFace on first run. Subsequent runs load from cache instantly.

---

## Author

**Sherwin Alle**  
M.S. Computer Science — California State University, Fresno  
[GitHub](https://github.com/Sherwin951) · [Email](mailto:alle.sherwin9999@gmail.com)
