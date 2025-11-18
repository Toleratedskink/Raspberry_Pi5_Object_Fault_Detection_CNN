"""
YOLOv8 Welding Fault Detection - Simplified Training Script
"""

from ultralytics import YOLO
from pathlib import Path

def train():
    """Train YOLOv8 model."""
    data_yaml = Path("data/weld_dataset.yaml")
    if not data_yaml.exists():
        print("Error: Run setup_kaggle_dataset.py first!")
        return
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=str(data_yaml),
        epochs=50,           # Reduced for faster training
        imgsz=640,
        batch=8,            # Smaller batch for limited dataset
        name='weld_detection',
        project="runs",
        patience=20,
        device='cpu',
    )
    
    print(f"\n✓ Training complete! Model: {results.save_dir}/weights/best.pt")
    return results

def validate(model_path=None):
    """Validate model."""
    model_path = model_path or "runs/weld_detection/weights/best.pt"
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return
    
    model = YOLO(model_path)
    results = model.val(data="data/weld_dataset.yaml")
    print(f"\nValidation Results: {results}")

def predict(image_path, model_path=None, conf=0.25):
    """Predict on an image."""
    model_path = model_path or "runs/weld_detection/weights/best.pt"
    model = YOLO(model_path)
    
    results = model.predict(image_path, conf=conf, save=True)
    
    for result in results:
        print(f"\nDetections in {image_path}:")
        if result.boxes:
            for box in result.boxes:
                print(f"  - {model.names[int(box.cls[0])]}: {float(box.conf[0]):.2%}")
        else:
            print("  No defects detected")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'validate', 'predict'], default='train')
    parser.add_argument('--model', help='Model path')
    parser.add_argument('--image', help='Image path for predict')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'validate':
        validate(args.model)
    elif args.mode == 'predict':
        if not args.image:
            print("Error: --image required for predict")
        else:
            predict(args.image, args.model, args.conf)
