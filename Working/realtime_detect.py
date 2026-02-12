"""
Real-time weld defect detection using YOLOv8 and a webcam or video file.

Author intent (student notes):
- This script runs the trained welding-defect model on a live camera feed or
  a video file and displays bounding boxes and labels in real time.
- Useful for demos or inspecting a weld via camera (e.g., Raspberry Pi + camera).

Usage:
  # Webcam (default camera, index 0)
  python3 realtime_detect.py

  # Webcam with custom model and confidence
  python3 realtime_detect.py --model runs/weld_detection4/weights/best.pt --conf 0.3

  # Video file
  python3 realtime_detect.py --source path/to/video.mp4

  # Save output video
  python3 realtime_detect.py --save output.mp4

Press 'q' in the display window to quit.

The model was trained on images of welds (cracks, porosity, spatter, etc.). To see
defect boxes, point the camera at an actual weld or at a photo of a weld (e.g.
on another screen or printed). Pointing at a room/face will correctly show "no defects".
Lower confidence with --conf 0.15 if you want more detections (more false positives).

macOS: If you see a black screen, grant camera access:
  System Settings → Privacy & Security → Camera → turn ON for Terminal (or Cursor).
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def _print_macos_camera_tip(source):
    """Print a short tip for macOS camera permission."""
    if source == 0 or source == "0":
        print("  → System Settings → Privacy & Security → Camera")
        print("  → Turn ON access for Terminal (or Cursor / VS Code)")


def run_realtime(
    source=0,
    model_path="runs/weld_detection4/weights/best.pt",
    conf=0.25,
    save_path=None,
):
    """
    Run YOLOv8 weld-defect detection on a video source (webcam or file).

    Parameters:
    - source: camera index (e.g., 0 for default webcam) or path to video file
    - model_path: path to trained YOLO weights (.pt)
    - conf: confidence threshold for detections (0–1)
    - save_path: if set, save annotated video to this path
    """
    # Load model once (reused for every frame)
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        print("Train a model first with: python3 main.py --mode train")
        return
    model = YOLO(model_path)

    # Open video source: 0 = default webcam, or path to .mp4 / device
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        _print_macos_camera_tip(source)
        return

    # On macOS, camera often returns "opened" but frames are black until permission is granted.
    # Try one test frame so we can warn the user.
    if source == 0 or source == "0":
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("Error: Could not read a frame from the camera.")
            _print_macos_camera_tip(source)
            cap.release()
            return
        if test_frame.size == 0 or test_frame.max() == 0:
            print("Camera returned an empty/black frame. On macOS, grant camera access:")
            _print_macos_camera_tip(source)
            cap.release()
            return
        # Reset so the main loop (YOLO) can read from the start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Optional: video writer for saving
    writer = None
    if save_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    window_name = "Weld Defect Detection (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if source == 0 or source == "0":
        print("Using default webcam. Point camera at a weld (or photo of a weld) to see defect boxes.")
        print("If the window is black, grant camera access:")
        _print_macos_camera_tip(source)

    try:
        # Stream inference: YOLO returns a generator when stream=True
        # source can be camera index (int) or path (str)
        fps_smooth = 0.0
        t_prev = time.perf_counter()
        for result in model.predict(
            source=source,
            stream=True,
            conf=conf,
            verbose=False,
            device="cpu",  # use "0" or "cuda" if GPU available
        ):
            # result.plot() returns the frame with boxes/labels drawn (BGR)
            frame = result.plot()

            # Show status: detections or "no defects" (model was trained on weld images)
            h, w = frame.shape[:2]
            if result.boxes is None or len(result.boxes) == 0:
                status = "No defects - point camera at weld (or weld image)"
                cv2.putText(
                    frame,
                    status,
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),  # orange
                    2,
                )
            else:
                n = len(result.boxes)
                cv2.putText(
                    frame,
                    f"Defects: {n}",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # FPS (smoothed)
            t = time.perf_counter()
            if fps_smooth == 0:
                fps_smooth = 1.0
            else:
                fps_smooth = 0.9 * fps_smooth + 0.1 / (t - t_prev)
            t_prev = t
            cv2.putText(
                frame,
                f"FPS: {fps_smooth:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if writer is not None:
                writer.write(frame)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time weld defect detection from webcam or video file."
    )
    parser.add_argument(
        "--source",
        default=0,
        help="Camera index (e.g., 0) or path to video file. Default: 0 (webcam)",
    )
    parser.add_argument(
        "--model",
        default="runs/weld_detection4/weights/best.pt",
        help="Path to YOLO model weights (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (0–1). Default: 0.25",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="If set, save annotated video to this path (e.g., output.mp4)",
    )
    args = parser.parse_args()

    # Allow numeric camera index from CLI (e.g., --source 0)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run_realtime(
        source=source,
        model_path=args.model,
        conf=args.conf,
        save_path=args.save,
    )
