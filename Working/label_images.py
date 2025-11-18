"""
Image Labeling Helper Script
This script helps create YOLO format annotations for weld images.
For a GUI tool, consider using labelImg: https://github.com/HumanSignal/labelImg
"""

import os
from pathlib import Path
import json

# Weld categories mapping
WELD_CATEGORIES = {
    "good_weld": 0,
    "porosity": 1,
    "crack": 2,
    "undercut": 3,
    "lack_of_fusion": 4,
    "slag_inclusion": 5,
    "spatter": 6,
}

def create_empty_label_file(image_path, label_dir):
    """
    Create an empty label file for an image (no defects detected).
    """
    image_path = Path(image_path)
    label_path = Path(label_dir) / f"{image_path.stem}.txt"
    label_path.write_text("")  # Empty file means no objects
    return label_path

def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """
    Convert bounding box from absolute coordinates to YOLO format.
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    
    Args:
        x_min, y_min, x_max, y_max: Bounding box coordinates
        img_width, img_height: Image dimensions
    
    Returns:
        Tuple of (center_x, center_y, width, height) normalized
    """
    # Calculate center and dimensions
    center_x = (x_min + x_max) / 2.0 / img_width
    center_y = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Ensure values are within [0, 1]
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height

def create_label_file(image_path, label_dir, annotations):
    """
    Create a YOLO format label file.
    
    Args:
        image_path: Path to image file
        label_dir: Directory to save label file
        annotations: List of dicts with keys: 'class', 'bbox' (x_min, y_min, x_max, y_max)
    """
    from PIL import Image
    
    image_path = Path(image_path)
    label_path = Path(label_dir) / f"{image_path.stem}.txt"
    
    # Get image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Write annotations
    with open(label_path, 'w') as f:
        for ann in annotations:
            class_name = ann['class']
            if class_name not in WELD_CATEGORIES:
                print(f"Warning: Unknown class '{class_name}', skipping...")
                continue
            
            class_id = WELD_CATEGORIES[class_name]
            bbox = ann['bbox']
            
            # Convert to YOLO format
            center_x, center_y, width, height = convert_bbox_to_yolo(
                bbox[0], bbox[1], bbox[2], bbox[3],
                img_width, img_height
            )
            
            # Write line: class_id center_x center_y width height
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Created label file: {label_path}")
    return label_path

def print_labeling_instructions():
    """
    Print instructions for labeling images.
    """
    print("\n" + "="*60)
    print("IMAGE LABELING INSTRUCTIONS")
    print("="*60)
    print("\nFor YOLOv8, you need to create label files (.txt) for each image.")
    print("\nLabel File Format (YOLO):")
    print("  class_id center_x center_y width height")
    print("  (all values normalized 0-1)")
    print("\nExample:")
    print("  1 0.5 0.5 0.3 0.2")
    print("  (class 1, centered at 50% width/height, 30% width, 20% height)")
    print("\nClass IDs:")
    for name, id in WELD_CATEGORIES.items():
        print(f"  {id}: {name}")
    print("\nRecommended Tools:")
    print("1. labelImg (GUI tool): https://github.com/HumanSignal/labelImg")
    print("   Install: pip install labelImg")
    print("   Run: labelImg")
    print("\n2. CVAT (web-based): https://github.com/openvinotoolkit/cvat")
    print("\n3. Manual: Edit .txt files directly using this script's functions")
    print("="*60)

if __name__ == "__main__":
    print_labeling_instructions()
    
    print("\nExample usage in code:")
    print("""
from label_images import create_label_file

# Example: Label an image with a porosity defect
annotations = [
    {
        'class': 'porosity',
        'bbox': [100, 150, 200, 250]  # x_min, y_min, x_max, y_max
    }
]

create_label_file(
    'data/images/train/weld1.jpg',
    'data/labels/train',
    annotations
)
    """)

