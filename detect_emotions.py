import sys
import time
import numpy as np
import cv2
from collections import deque
from PIL import Image
from transformers import pipeline

EMOTION_COLORS = {
    "happy":    (0, 220, 0),
    "sad":      (220, 50, 0),
    "angry":    (0, 0, 220),
    "fear":     (180, 0, 180),
    "surprise": (0, 200, 230),
    "disgust":  (0, 130, 220),
    "neutral":  (220, 220, 220),
}

EMOTIONS_ORDER = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_model():
    print("Loading emotion model (first run downloads ~300MB)...")
    pipe = pipeline(
        "image-classification",
        model="trpakov/vit-face-expression",
        device=-1,
    )
    print("Model ready.\n")
    return pipe


def classify_face(pipe, face_img_bgr: np.ndarray) -> tuple[str, dict]:
    """Run emotion classification on a cropped face (BGR numpy array)."""
    rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize((224, 224))
    results = pipe(pil_img, top_k=7)
    scores = {r["label"].lower(): round(r["score"] * 100, 2) for r in results}
    dominant = max(scores, key=scores.get)
    return dominant, scores


def get_dominant_from_window(window: deque) -> str:
    if not window:
        return "neutral"
    counts: dict[str, int] = {}
    for e in window:
        counts[e] = counts.get(e, 0) + 1
    return max(counts, key=counts.get)


def draw_overlay(frame: np.ndarray, dominant: str, scores: dict, fps: float, face_rect=None) -> np.ndarray:
    h, w = frame.shape[:2]

    # Dark header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = EMOTION_COLORS.get(dominant, (255, 255, 255))
    cv2.putText(frame, dominant.upper(), (15, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

    # Draw face rectangle
    if face_rect is not None:
        x, y, fw, fh = face_rect
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

    # Emotion probability bars (right side)
    bar_x = w - 200
    bar_w = 160
    bar_h = 12
    for i, emo in enumerate(EMOTIONS_ORDER):
        score = scores.get(emo, 0.0) / 100.0
        y_pos = 20 + i * 22
        emo_color = EMOTION_COLORS.get(emo, (200, 200, 200))
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * score)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + fill_w, y_pos + bar_h), emo_color, -1)
        cv2.putText(frame, f"{emo[:4]} {score*100:.0f}%", (bar_x - 90, y_pos + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, emo_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q: quit  S: screenshot", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    return frame


def analyze_image(image_path: str) -> None:
    """Analyze emotions in a static image and display results."""
    pipe = load_model()
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: could not read image at {image_path}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        print("No face detected — analyzing full image.")
        dominant, scores = classify_face(pipe, frame)
        face_rect = None
    else:
        x, y, fw, fh = faces[0]
        face_crop = frame[y:y+fh, x:x+fw]
        dominant, scores = classify_face(pipe, face_crop)
        face_rect = (x, y, fw, fh)

    print(f"\nDominant emotion: {dominant.upper()}")
    for emo in EMOTIONS_ORDER:
        bar = "█" * int(scores.get(emo, 0) / 5)
        print(f"  {emo:<10}: {bar:<20} {scores.get(emo, 0):.1f}%")

    annotated = draw_overlay(frame.copy(), dominant, scores, 0.0, face_rect)
    out_path = image_path.rsplit(".", 1)[0] + "_result.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\nSaved: {out_path}")
    cv2.imshow("Emotion Analysis", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam() -> None:
    pipe = load_model()

    print("=== Real-Time Emotion Detector ===")
    print("Controls: Q = quit | S = save screenshot\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: webcam not found. Try changing VideoCapture(0) to VideoCapture(1).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    emotion_window: deque = deque(maxlen=5)
    last_scores: dict = {e: 0.0 for e in EMOTIONS_ORDER}
    last_dominant = "neutral"
    last_face_rect = None
    screenshot_count = 0
    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    frame_idx = 0
    ANALYSIS_INTERVAL = 5  # analyze every 5th frame for smooth FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_idx += 1
        frame_count += 1
        now = time.time()
        elapsed = now - prev_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = now

        if frame_idx % ANALYSIS_INTERVAL == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

            if len(faces) > 0:
                x, y, fw, fh = faces[0]
                last_face_rect = (x, y, fw, fh)
                try:
                    face_crop = frame[y:y+fh, x:x+fw]
                    dominant, scores = classify_face(pipe, face_crop)
                    last_dominant = dominant
                    last_scores = scores
                    emotion_window.append(dominant)
                except Exception:
                    pass
            else:
                last_face_rect = None

        smoothed = get_dominant_from_window(emotion_window) if emotion_window else last_dominant
        display = draw_overlay(frame.copy(), smoothed, last_scores, fps, last_face_rect)
        cv2.imshow("Emotion Detector — Press Q to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            screenshot_count += 1
            fname = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(fname, display)
            print(f"Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_image(sys.argv[1])
    else:
        run_webcam()
