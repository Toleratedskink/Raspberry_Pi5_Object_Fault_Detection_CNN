# YOLOv8 Welding Defect Detection - Setup and Usage Guide

This project uses YOLOv8 (Ultralytics) to detect welding defects in images. This README explains how to set up the environment, install dependencies, and run training, validation, and prediction.

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

---

## Initial Setup (First Time Only)

### Step 1: Create Virtual Environment

Navigate to the `Working` directory and create a Python virtual environment:

```bash
cd /Users/jeremyburke/Raspberry_Pi5_Object_Fault_Detection_CNN/Working
python3 -m venv venv
```

This creates a `venv` folder that will contain all project-specific Python packages.

### Step 2: Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**Alternative (using the provided script):**
```bash
source activate.sh
```

You should see `(venv)` appear at the beginning of your terminal prompt, indicating the environment is active.

### Step 3: Install Required Packages

With the virtual environment activated, install all dependencies:

```bash
pip install -r requirements.txt
```

This installs:
- `ultralytics` (YOLOv8 framework)
- `torch` and `torchvision` (PyTorch for deep learning)
- `opencv-python` (image processing)
- And other supporting libraries

**Note:** Installation may take several minutes, especially PyTorch.

---

## Daily Usage

### Activating the Environment

Every time you open a new terminal session, activate the virtual environment:

```bash
cd /Users/jeremyburke/Raspberry_Pi5_Object_Fault_Detection_CNN/Working
source venv/bin/activate
```

Or use the convenience script:
```bash
source activate.sh
```

### Deactivating the Environment

When you're done working, you can deactivate:
```bash
deactivate
```

---

## Running the Script

**Important:** Always use `python3 main.py` (not just `main.py`). The script must be run with Python.

### Training a Model

Train a YOLOv8 model using the pictures in `data/` (train/val images and labels):

```bash
python3 train.py
```

Or with options:

```bash
python3 train.py --epochs 100 --batch 16 --name my_run
python3 train.py --device 0
```

**What this does:**
- Uses the dataset defined in `data/weld_dataset.yaml` (images in `data/images/train`, `data/images/val`)
- Starts from a pretrained YOLOv8n (nano) model and fine-tunes on your weld images
- Saves the best weights to `runs/weld_detection/weights/best.pt` (or `runs/<name>/weights/best.pt` if you use `--name`)

**Optional arguments:** `--data`, `--base`, `--epochs`, `--imgsz`, `--batch`, `--name`, `--project`, `--patience`, `--device`. Run `python3 train.py --help` for details.

**Note:** Prepare the dataset first with `python3 setup_kaggle_dataset.py` if you haven’t already. You can also train via `python3 main.py --mode train` (same behavior).

**Training with the large dataset:** The folder **The Welding Defect Dataset** (project root) has many more images. Use it with: `python3 train.py --data weld_big_dataset.yaml --name weld_big` (3 classes: Bad Weld, Good Weld, Defect). For good/bad-only: run `python3 prepare_2class_big_dataset.py`, then `python3 train.py --data data_2class_big/weld_2class_big.yaml --name weld_good_bad`.

### Validating a Model

Evaluate a trained model on the validation set:

```bash
python3 main.py --mode validate
```

To validate a specific model (from a different training run):
```bash
python3 main.py --mode validate --model runs/weld_detection4/weights/best.pt
```

**What this does:**
- Loads the trained model weights
- Runs inference on the validation images
- Prints evaluation metrics (mAP, precision, recall, etc.)

### Running Predictions

Detect defects in a single image:

```bash
python3 main.py --mode predict \
  --image data/images/test/crack-welding-images_17_jpeg_jpg.rf.ff570716da313da84983bc629ae7e331.jpg \
  --model runs/weld_detection4/weights/best.pt \
  --conf 0.25
```

**Parameters:**
- `--image`: Path to the image file you want to analyze
- `--model`: Path to the model weights file (`.pt` file). If omitted, defaults to `runs/weld_detection/weights/best.pt`
- `--conf`: Confidence threshold (0.0 to 1.0). Higher = fewer but more confident detections. Default: 0.25

**What this does:**
- Loads the trained model
- Runs inference on the specified image
- Prints detected defects to the terminal
- Saves an annotated image (with bounding boxes) to `runs/detect/predict/`

**Example output:**
```
Detections in image.jpg:
  - crack: 85.3%
  - porosity: 72.1%
```

### Real-time detection (webcam or video)

Run weld-defect detection on a live camera feed or a video file:

```bash
# Default webcam (camera index 0)
python3 realtime_detect.py

# Custom model and confidence
python3 realtime_detect.py --model runs/weld_detection4/weights/best.pt --conf 0.3

# Use a video file instead of webcam
python3 realtime_detect.py --source path/to/video.mp4

# Save annotated video to a file
python3 realtime_detect.py --save output.mp4
```

**Parameters:**
- `--source`: Camera index (e.g., `0` for default webcam) or path to a video file. Default: `0`
- `--model`: Path to trained YOLO weights (`.pt`). Default: `runs/weld_detection4/weights/best.pt`
- `--conf`: Confidence threshold (0–1). Default: `0.25`
- `--save`: If set, save the annotated video to this path (e.g., `output.mp4`)

**What this does:**
- Opens your webcam or the given video file
- Runs the trained model on each frame and draws bounding boxes and labels
- Shows FPS in the window
- Press **q** in the window to quit

### Real-time Good / Bad only (2-class model)

Uses a model trained on **only two classes**: good_weld and bad_weld (no defect types). Setup once, then run realtime:

**One-time setup:** Run `python3 prepare_2class_dataset.py`, then train the 2-class model:
`python3 train.py --data data_2class/weld_dataset_2class.yaml --name weld_good_bad`

**Run realtime:** `python3 realtime_good_bad.py` or `python3 realtime_good_bad.py --conf 0.2 --save output.mp4`

- **GOOD WELD** (green) / **BAD WELD** (red). Boxes show confidence only.
Press **q** to quit.

---

## Quick Reference

| Task | Command |
|------|---------|
| Activate environment | `source venv/bin/activate` |
| Install packages | `pip install -r requirements.txt` |
| Train model | `python3 train.py` or `python3 main.py --mode train` |
| Validate model | `python3 main.py --mode validate` |
| Predict on image | `python3 main.py --mode predict --image <path> --model <path> --conf 0.25` |
| **Real-time (full)** | `python3 realtime_detect.py` or `python3 realtime_detect.py --source video.mp4` |
| **Real-time (good/bad only)** | `python3 realtime_good_bad.py` |
| Get help | `python3 main.py --help` |

---

## Troubleshooting

### "python: command not found"
- Use `python3` instead of `python` on macOS

### "No module named 'ultralytics'"
- Make sure the virtual environment is activated (`venv` should appear in your prompt)
- Run `pip install -r requirements.txt` again

### "Model not found" error
- Check that training has completed successfully
- Verify the model path: `ls runs/weld_detection*/weights/best.pt`
- Use `--model` flag to specify the correct path

### "FileNotFoundError" for images
- Verify the image path is correct: `ls data/images/test/`
- Use tab completion to avoid typos in long filenames

---

## Project Structure

```
Working/
├── main.py                 # Main training/validation/prediction script
├── train.py                # Dedicated training script (uses data/ pictures)
├── realtime_detect.py      # Real-time weld defect detection (webcam/video, full labels)
├── realtime_good_bad.py    # Real-time good/bad only (2-class model: good_weld / bad_weld)
├── prepare_2class_dataset.py  # Converts 7-class labels to 2-class for good/bad training
├── setup_kaggle_dataset.py # Dataset preparation script
├── requirements.txt        # Python package dependencies
├── activate.sh            # Convenience script to activate venv
├── data/                  # Dataset (images and labels)
│   └── weld_dataset.yaml  # Dataset configuration file
├── venv/                  # Virtual environment (created during setup)
└── runs/                  # Training outputs and predictions
    ├── weld_detection*/   # Training run directories
    └── detect/            # Prediction output images
```

---

## Notes for Reviewers

- This project uses transfer learning: starting from a pretrained YOLOv8n model and fine-tuning on a custom welding defect dataset
- Training configuration is set for CPU by default (`device='cpu'`). Change to `'0'` or `'cuda'` in `main.py` if GPU is available
- The dataset must be prepared using `setup_kaggle_dataset.py` before training
- Model weights are saved automatically during training; the best-performing model is saved as `best.pt`
