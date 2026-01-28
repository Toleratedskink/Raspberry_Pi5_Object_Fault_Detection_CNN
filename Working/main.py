"""
YOLOv8 Welding Fault Detection - Simplified Training / Validation / Inference Script

Author intent (student notes):
- This file is meant to be a *single entry point* for my welding-defect detection project.
- It wraps Ultralytics YOLOv8's Python API so I can reproducibly:
  1) train a model on my dataset,
  2) validate the trained model, and
  3) run predictions on new images.

Project assumptions:
- A YOLO-format dataset has been prepared already (images + labels).
- A dataset config file exists at: `data/weld_dataset.yaml`
  This YAML tells YOLO where the train/val images live and what the class names are.
- Training outputs are written under `runs/` (Ultralytics default convention).

How to run (examples):
- Train:
    python main.py --mode train
- Validate:
    python main.py --mode validate --model runs/weld_detection/weights/best.pt
- Predict:
    python main.py --mode predict --image path/to/image.jpg --conf 0.25

Notes:
- This script is intentionally "simple and explicit" (not optimized) so a professor can
  understand the pipeline quickly.
"""

from ultralytics import YOLO
from pathlib import Path

def train():
    """
    Train a YOLOv8 model on the welding defect dataset.

    Key idea:
    - We start from a pretrained YOLOv8 checkpoint (`yolov8n.pt`) and fine-tune it
      on our custom welding dataset. This is transfer learning and is typical when
      the dataset is not huge.
    """
    # The dataset YAML is the "contract" between our dataset folder structure
    # and the Ultralytics training code.
    data_yaml = Path("data/weld_dataset.yaml")
    if not data_yaml.exists():
        # Early-exit guard: training cannot start without the dataset config.
        # This message points to the script that prepares/downloads the dataset.
        print("Error: Run setup_kaggle_dataset.py first!")
        return
    
    # Load a YOLOv8 model. 'yolov8n.pt' is the "nano" model (smallest/fastest).
    # Using a smaller model is helpful for faster iteration and for limited hardware.
    model = YOLO('yolov8n.pt')
    
    # Train configuration:
    # - epochs: number of full passes over the training data
    # - imgsz: input image size YOLO resizes to (square)
    # - batch: images per optimization step (smaller batch uses less RAM)
    # - name/project: where outputs are saved (runs/<project>/<name>/...)
    # - patience: early stopping patience (stop if val doesn't improve)
    # - device: 'cpu' here for compatibility; change to '0' or 'cuda' if GPU exists
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
    
    # `results.save_dir` is an Ultralytics path object pointing to the run directory.
    # The best-performing weights (by validation metric) are stored in weights/best.pt.
    print(f"\n✓ Training complete! Model: {results.save_dir}/weights/best.pt")
    return results

def validate(model_path=None):
    """
    Validate a trained model on the validation split.

    Why validation matters:
    - It provides metrics (e.g., mAP) that quantify detection performance on data the
      model did not train on.
    - It helps detect overfitting and compare training runs objectively.

    Parameters:
    - model_path: path to a YOLO weights file (.pt). If None, defaults to the "best"
      weights from our standard runs folder.
    """
    # Default path assumes the same `project="runs"` and `name="weld_detection"` used in train().
    model_path = model_path or "runs/weld_detection/weights/best.pt"
    if not Path(model_path).exists():
        # Helpful guard so we don't crash if training hasn't been run yet.
        print(f"Model not found: {model_path}")
        return
    
    # Load the trained model weights.
    model = YOLO(model_path)
    # Run validation against the same dataset YAML (it encodes the val split location).
    results = model.val(data="data/weld_dataset.yaml")
    # Printing `results` gives a summary of evaluation metrics.
    print(f"\nValidation Results: {results}")

def predict(image_path, model_path=None, conf=0.25):
    """
    Run inference (prediction) on a single image and print detected defects.

    Parameters:
    - image_path: file path to the image to analyze
    - model_path: weights file (.pt) to use; defaults to the project's best weights
    - conf: confidence threshold in [0, 1]; higher means fewer (but more confident)
      detections. Lower means more detections (but may include false positives).
    """
    # Default model path matches validate() and assumes a prior training run.
    model_path = model_path or "runs/weld_detection/weights/best.pt"
    # Load model weights for inference.
    model = YOLO(model_path)
    
    # Run prediction.
    # - save=True writes output images with drawn boxes into a `runs/` subfolder.
    # - results is a list (even for one image) of Ultralytics Results objects.
    results = model.predict(image_path, conf=conf, save=True)
    
    # Print detections to the terminal in a human-readable way.
    for result in results:
        print(f"\nDetections in {image_path}:")
        if result.boxes:
            for box in result.boxes:
                # Each `box` contains:
                # - cls: predicted class index
                # - conf: confidence score
                # `model.names` maps class index -> string label from the dataset YAML.
                print(f"  - {model.names[int(box.cls[0])]}: {float(box.conf[0]):.2%}")
        else:
            print("  No defects detected")

if __name__ == "__main__":
    # Command-line interface (CLI):
    # This lets me run the same Python file in different "modes" without editing code.
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train/validate/predict a YOLOv8 model for welding defect detection."
    )
    # Choose what action to perform.
    parser.add_argument('--mode', choices=['train', 'validate', 'predict'], default='train')
    # Optional model path override (useful for validating/predicting with a specific run).
    parser.add_argument('--model', help='Model path')
    # Only used for prediction mode (path to a single image).
    parser.add_argument('--image', help='Image path for predict')
    # Confidence threshold for prediction mode.
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    # Parse args from the shell command.
    args = parser.parse_args()
    
    # Dispatch based on requested mode.
    if args.mode == 'train':
        train()
    elif args.mode == 'validate':
        validate(args.model)
    elif args.mode == 'predict':
        if not args.image:
            # For predict we require an image path; otherwise inference can't run.
            print("Error: --image required for predict")
        else:
            predict(args.image, args.model, args.conf)
