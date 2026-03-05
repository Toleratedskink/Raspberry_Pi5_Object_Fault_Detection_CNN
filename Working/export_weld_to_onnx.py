"""
Export the 2-class weld model to ONNX (opset=11) for Hailo-8 compilation.

Run from Working/ with venv activated. Install ONNX first if needed: pip install onnx
  python3 export_weld_to_onnx.py
  python3 export_weld_to_onnx.py --model runs/weld_good_bad2/weights/best.pt --out weld_2class.onnx

Then compile to HEF on a machine with Hailo Dataflow Compiler, e.g.:
  hailomz compile --ckpt weld_2class.onnx --calib-path data_2class_big/images/valid \\
    --yaml yolov8n.yaml --classes 2 --hw-arch hailo8l
(Use the YAML from Hailo Model Zoo for your backbone, e.g. yolov8n for nano.)
"""

import argparse
import sys
from pathlib import Path

try:
    import onnx  # noqa: F401
except ImportError:
    print("ONNX not found. Install with: pip install onnx onnxruntime onnxslim")
    sys.exit(1)

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Export weld 2-class model to ONNX for Hailo.")
    ap.add_argument("--model", default="runs/weld_good_bad2/weights/best.pt", help="Path to .pt model")
    ap.add_argument("--out", default="weld_2class.onnx", help="Output ONNX path")
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        return
    model = YOLO(args.model)
    # opset=11 required for Hailo compilation; output is next to .pt by default
    out_path = model.export(format="onnx", opset=11, imgsz=640, simplify=True)
    if args.out and Path(out_path).resolve() != Path(args.out).resolve():
        import shutil
        shutil.copy(out_path, args.out)
        print(f"Exported and copied to: {args.out}")
    else:
        print(f"Exported: {out_path}")
    print("Next: compile to HEF with Hailo Dataflow Compiler (see HAILO.md).")


if __name__ == "__main__":
    main()
