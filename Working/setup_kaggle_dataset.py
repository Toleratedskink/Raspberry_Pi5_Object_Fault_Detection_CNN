"""
Simple dataset setup for welding defect detection.
Organizes local dataset into YOLOv8 format.
"""

import shutil
from pathlib import Path
import yaml
import random

# Categories
CATEGORIES = ["good_weld", "porosity", "crack", "undercut", "lack_of_fusion", "slag_inclusion", "spatter"]

def organize_dataset(source_dir, output_dir="data", max_per_category=2):
    """Organize dataset into YOLOv8 format with limited images per category."""
    source = Path(source_dir)
    output = Path(output_dir)
    
    # Create structure
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Find splits (handle both "val" and "valid")
    splits = {}
    for split_name in ["train", "val", "valid", "test"]:
        split_path = source / split_name
        if split_path.exists():
            splits[split_name if split_name != "valid" else "val"] = split_path
    
    # Copy files with limit
    random.seed(42)
    for split_name, split_path in splits.items():
        img_dir = split_path / "images"
        lbl_dir = split_path / "labels"
        
        if not img_dir.exists():
            continue
        
        # Get all images, limit per category
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        random.shuffle(images)
        
        # Group by category and limit
        category_images = {}
        for img in images:
            # Try to infer category from filename
            for cat in CATEGORIES:
                if cat.replace("_", "") in img.stem.lower():
                    if cat not in category_images:
                        category_images[cat] = []
                    category_images[cat].append(img)
                    break
        
        # Copy limited images
        copied = 0
        for cat, imgs in category_images.items():
            for img in imgs[:max_per_category]:
                # Copy image
                shutil.copy(img, output / "images" / split_name / img.name)
                
                # Copy label if exists
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy(lbl, output / "labels" / split_name / lbl.name)
                else:
                    (output / "labels" / split_name / f"{img.stem}.txt").write_text("")
                copied += 1
        
        print(f"  {split_name}: {copied} images")
    
    # Create YAML config
    config = {
        'path': str(output.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CATEGORIES),
        'names': CATEGORIES
    }
    
    with open(output / "weld_dataset.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✓ Dataset ready: {output}")
    return output

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup welding defect dataset')
    parser.add_argument('--local', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--max', type=int, default=2, help='Max images per category (default: 2)')
    parser.add_argument('--yes', '-y', action='store_true', help='Overwrite existing data')
    
    args = parser.parse_args()
    
    if Path("data").exists() and not args.yes:
        response = input("'data' exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            exit()
        shutil.rmtree("data")
    
    print(f"Organizing dataset (max {args.max} images per category)...")
    organize_dataset(args.local, max_per_category=args.max)
