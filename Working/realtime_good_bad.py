"""
Real-time Good Weld / Bad Weld only.

Uses a model trained with exactly two classes: good_weld (0) and bad_weld (1).
No defect types (no crack, porosity, spatter, etc.) — only GOOD WELD or BAD WELD.

Setup (one-time):
  1. python3 prepare_2class_dataset.py
  2. python3 train.py --data data_2class_big/weld_2class_big.yaml --name weld_good_bad2
  3. python3 realtime_good_bad.py

Usage:
  python3 realtime_good_bad.py
  python3 realtime_good_bad.py --conf 0.2 --imgsz 320
  # Raspberry Pi 5: default source /dev/video0. For Hailo-8, see HAILO.md and run_realtime_hailo.py.
Press 'q' to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# 2-class model: 0 = good_weld, 1 = bad_weld
GOOD_CLASS_ID = 0


def _macos_camera_tip():
    print("  → System Settings → Privacy & Security → Camera")
    print("  → Turn ON for Terminal (or Cursor)")


def run(source="/dev/video0", model_path="runs/weld_good_bad2/weights/best.pt", conf=0.25, bad_min_conf=0.5, save_path=None, imgsz=320, skip_frames=0):
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("  Train first, e.g.: python3 train.py --data data_2class_big/weld_2class_big.yaml --name weld_good_bad2")
        return
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source}")
        _macos_camera_tip()
        return

    # Camera source: /dev/video0 (Raspberry Pi) or 0 (e.g. macOS webcam)
    is_camera = source in (0, "0", "/dev/video0", "/dev/video1")
    if is_camera:
        ret, test = cap.read()
        if not ret or test is None or test.size == 0 or test.max() == 0:
            print("Could not read from camera. On macOS grant camera access; on Pi check /dev/video0.")
            _macos_camera_tip()
            cap.release()
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = None
    if save_path:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Use our camera only — read frames and pass to model (keeps camera open on Pi)
    win = "Good / Bad Weld (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if is_camera:
        print("Point camera at a weld. Green = GOOD WELD, Red = BAD WELD. Press 'q' to quit.")

    fps_smooth = 0.0
    t_prev = time.perf_counter()
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            h, w = frame.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 4
            frame_count += 1
            # Skip frames to reduce load (e.g. run inference every 2nd frame)
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Run model (smaller imgsz = faster on CPU; use --imgsz 640 for accuracy)
            results = model.predict(frame, conf=conf, verbose=False, device="cpu", imgsz=imgsz)
            result = results[0] if results else None
            if result is None:
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Only call it BAD if we have a bad_weld detection above bad_min_conf
            is_bad = False
            max_conf = 0.0
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    c = float(box.conf[0])
                    max_conf = max(max_conf, c)
                    if cls_id != GOOD_CLASS_ID and c >= bad_min_conf:
                        is_bad = True
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
                    color = (0, 0, 255) if cls_id != GOOD_CLASS_ID else (0, 255, 0)
                    cv2.rectangle(frame, pt1, pt2, color, thickness)
                    txt = f"{c:.0%}"
                    (cw, ch), _ = cv2.getTextSize(txt, font, 0.6, 2)
                    cv2.rectangle(frame, (pt1[0], pt1[1] - ch - 6), (pt1[0] + cw + 4, pt1[1]), color, -1)
                    cv2.putText(frame, txt, (pt1[0] + 2, pt1[1] - 4), font, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(
                    frame, f"No detections (conf={conf}). Try --conf 0.15",
                    (10, h - 12), font, 0.45, (0, 165, 255), 1,
                )

            verdict = "BAD WELD" if is_bad else "GOOD WELD"
            color = (0, 0, 255) if is_bad else (0, 255, 0)
            if result.boxes is not None and len(result.boxes) > 0:
                verdict = f"{verdict} ({max_conf:.0%})"
            (tw, th), _ = cv2.getTextSize(verdict, font, 2.0, thickness)
            cx = (w - tw) // 2
            cy = (h + th) // 2
            cv2.putText(frame, verdict, (cx, cy), font, 2.0, color, thickness)

            t = time.perf_counter()
            fps_smooth = 0.9 * fps_smooth + 0.1 / (t - t_prev) if fps_smooth else 1.0
            t_prev = t
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30), font, 1, (0, 255, 0), 2)

            if writer is not None:
                writer.write(frame)
            cv2.imshow(win, frame)
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
    ap = argparse.ArgumentParser(description="Real-time Good/Bad weld only (2-class model).")
    ap.add_argument("--source", default="/dev/video0", help="Camera device (default /dev/video0 for Raspberry Pi) or video path")
    ap.add_argument("--model", default="runs/weld_good_bad2/weights/best.pt", help="Path to 2-class .pt model")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    ap.add_argument("--bad-min-conf", type=float, default=0.5, help="Min confidence to show BAD WELD (higher = fewer false bads)")
    ap.add_argument("--imgsz", type=int, default=320, help="Inference size (320=faster, 640=more accurate). Use 320 on Pi CPU.")
    ap.add_argument("--skip-frames", type=int, default=0, help="Run inference every N+1 frames (0=every frame, 1=every 2nd, 2=every 3rd)")
    ap.add_argument("--save", default=None, help="Save output video path")
    args = ap.parse_args()
    # Allow numeric camera index (e.g. 0) or device path (/dev/video0)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    run(source=source, model_path=args.model, conf=args.conf, bad_min_conf=args.bad_min_conf, save_path=args.save, imgsz=args.imgsz, skip_frames=args.skip_frames)
