"""
Training script for the weld defect model — trains using the pictures in data/
(train/val images and labels). Uses the same dataset as main.py (data/weld_dataset.yaml).

Run from the Working directory with the venv activated:

  python3 train.py
  python3 train.py --epochs 100 --batch 16 --device 0
  python3 train.py --name my_run --epochs 30

Output: runs/<name>/weights/best.pt and last.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml="data/weld_dataset.yaml",
    base_model="yolov8n.pt",
    epochs=50,
    imgsz=640,
    batch=8,
    name="weld_detection",
    project="runs",
    patience=20,
    device="cpu",
):
    """
    Train YOLOv8 on the weld dataset (images in data/images/train, data/images/val).
    """
    data_path = Path(data_yaml)
    if not data_path.exists():
        print("Error: Dataset config not found:", data_path)
        print("  Prepare the dataset first: python3 setup_kaggle_dataset.py")
        return None

    print("Loading base model:", base_model)
    model = YOLO(base_model)

    print("Training on:", data_path)
    print("  Epochs:", epochs, "  Batch:", batch, "  Image size:", imgsz, "  Device:", device)
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project=project,
        patience=patience,
        device=device,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print("\n✓ Training complete!")
    print("  Best weights:", best)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train weld defect model using data/ images.")
    parser.add_argument("--data", default="data/weld_dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--base", default="yolov8n.pt", help="Base YOLO model (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--name", default="weld_detection", help="Run name (saved under runs/<name>)")
    parser.add_argument("--project", default="runs", help="Project folder for runs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", default="cpu", help="Device: cpu, 0, cuda")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        base_model=args.base,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        patience=args.patience,
        device=args.device,
    )
