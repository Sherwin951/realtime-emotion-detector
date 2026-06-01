import sys
import time
import numpy as np
import cv2
from collections import deque
from deepface import DeepFace

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


def get_dominant_from_window(window: deque) -> str:
    """Return the most common dominant emotion across recent frames."""
    if not window:
        return "neutral"
    counts: dict[str, int] = {}
    for e in window:
        counts[e] = counts.get(e, 0) + 1
    return max(counts, key=counts.get)


def draw_overlay(frame: np.ndarray, dominant: str, emotion_scores: dict, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = EMOTION_COLORS.get(dominant, (255, 255, 255))
    cv2.putText(frame, dominant.upper(), (15, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

    bar_x = w - 200
    bar_w = 160
    bar_h = 12
    for i, emo in enumerate(EMOTIONS_ORDER):
        score = emotion_scores.get(emo, 0.0) / 100.0
        y_pos = 20 + i * 22
        emo_color = EMOTION_COLORS.get(emo, (200, 200, 200))

        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * score)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + fill_w, y_pos + bar_h), emo_color, -1)
        cv2.putText(frame, f"{emo[:4]} {score*100:.0f}%", (bar_x - 85, y_pos + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, emo_color, 1, cv2.LINE_AA)

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 110, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q: quit  S: screenshot", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


def analyze_image(image_path: str) -> None:
    """Analyze emotions in a static image file and display results."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: could not read image at {image_path}")
        return

    print(f"Analyzing image: {image_path}")
    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
        )
        if isinstance(result, list):
            result = result[0]
        dominant = result["dominant_emotion"]
        scores = result["emotion"]
        print(f"Dominant emotion: {dominant.upper()}")
        print("All scores:")
        for emo in EMOTIONS_ORDER:
            print(f"  {emo:<10}: {scores.get(emo, 0.0):.2f}%")

        dummy_deque = deque([dominant], maxlen=5)
        annotated = draw_overlay(frame.copy(), dominant, scores, 0.0)
        out_path = image_path.rsplit(".", 1)[0] + "_emotion_result.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"Saved annotated image: {out_path}")

        cv2.imshow("Emotion Analysis", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as exc:
        print(f"Analysis failed: {exc}")


def run_webcam() -> None:
    print("\n=== Real-Time Emotion Detector ===")
    print("Controls: Q = quit | S = save screenshot")
    print("Warming up camera...\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (index 0). Try a different index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    emotion_window: deque = deque(maxlen=5)
    last_scores: dict = {e: 0.0 for e in EMOTIONS_ORDER}
    last_dominant = "neutral"
    screenshot_count = 0

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    # Only run DeepFace every N frames to keep UI responsive
    ANALYSIS_INTERVAL = 3
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
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
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                )
                if isinstance(result, list):
                    result = result[0]
                last_dominant = result["dominant_emotion"]
                last_scores = result["emotion"]
                emotion_window.append(last_dominant)
            except Exception:
                pass

        smoothed_dominant = get_dominant_from_window(emotion_window) if emotion_window else last_dominant
        display = draw_overlay(frame.copy(), smoothed_dominant, last_scores, fps)

        cv2.imshow("Emotion Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Exiting.")
            break
        elif key == ord("s"):
            screenshot_count += 1
            fname = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(fname, display)
            print(f"Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_image(sys.argv[1])
    else:
        run_webcam()
