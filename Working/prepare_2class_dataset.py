"""
Convert the 7-class weld dataset to 2 classes only: good_weld (0) and bad_weld (1).
- Class 0 (good_weld) stays 0.
- Classes 1–6 (porosity, crack, undercut, lack_of_fusion, slag_inclusion, spatter) become 1 (bad_weld).

Creates data_2class/ with symlinked images and new label files. Run once before
training the good/bad-only model:

  python3 prepare_2class_dataset.py

Then train with: python3 train.py --data data_2class/weld_dataset_2class.yaml --name weld_good_bad
"""

from pathlib import Path

DATA = Path("data")
DATA_2CLASS = Path("data_2class")
SPLITS = ("train", "val", "test")


def convert_label_line(line: str) -> str:
    """Convert one YOLO line: class 0 stays 0, classes 1-6 become 1."""
    parts = line.strip().split()
    if not parts:
        return ""
    try:
        cls = int(parts[0])
        new_cls = 0 if cls == 0 else 1
        return f"{new_cls} " + " ".join(parts[1:]) + "\n"
    except (ValueError, IndexError):
        return ""


def main():
    DATA.mkdir(exist_ok=True)
    DATA_2CLASS.mkdir(exist_ok=True)

    # Symlink image dirs so we don't copy pixels
    (DATA_2CLASS / "images").mkdir(exist_ok=True)
    for split in SPLITS:
        src = DATA / "images" / split
        dst = DATA_2CLASS / "images" / split
        if not src.exists():
            continue
        if dst.exists():
            if not dst.is_symlink():
                print(f"  Skip (exists, not symlink): {dst}")
            continue
        dst.symlink_to(src.resolve())
        print(f"  Linked images: {dst} -> {src}")

    # Convert labels
    for split in SPLITS:
        src_dir = DATA / "labels" / split
        dst_dir = DATA_2CLASS / "labels" / split
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for txt in src_dir.glob("*.txt"):
            lines = []
            with open(txt) as f:
                for line in f:
                    converted = convert_label_line(line)
                    if converted:
                        lines.append(converted)
            out = dst_dir / txt.name
            with open(out, "w") as f:
                f.writelines(lines)
            print(f"  Converted: {txt.name} -> {out}")

    # Write 2-class YAML (path relative to Working dir)
    yaml_path = DATA_2CLASS / "weld_dataset_2class.yaml"
    abs_path = yaml_path.resolve()
    content = f"""# 2-class weld dataset: good_weld (0) and bad_weld (1) only
names:
  - good_weld
  - bad_weld
nc: 2
path: {abs_path.parent}
train: images/train
val: images/val
test: images/test
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  Wrote: {yaml_path}")
    print("\nDone. Train with:")
    print("  python3 train.py --data data_2class/weld_dataset_2class.yaml --name weld_good_bad")


if __name__ == "__main__":
    main()
