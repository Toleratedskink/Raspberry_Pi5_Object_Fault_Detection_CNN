"""
Build a 2-class (good_weld / bad_weld) version of "The Welding Defect Dataset"
so you can train the good/bad-only model on the large set.

Original classes: 0=Bad Weld, 1=Good Weld, 2=Defect
Converted: good_weld=0 (was 1), bad_weld=1 (was 0 or 2)

Creates: data_2class_big/ with symlinked images and converted labels.
Run once, then: python3 train.py --data data_2class_big/weld_2class_big.yaml --name weld_good_bad
"""

from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "The Welding Defect Dataset"
DEST = Path(__file__).resolve().parent / "data_2class_big"
SPLITS = ("train", "valid", "test")


def convert_line(line: str) -> str:
    parts = line.strip().split()
    if not parts:
        return ""
    try:
        cls = int(parts[0])
        # 0=Bad, 1=Good, 2=Defect -> good_weld=0, bad_weld=1
        new_cls = 0 if cls == 1 else 1
        return f"{new_cls} " + " ".join(parts[1:]) + "\n"
    except (ValueError, IndexError):
        return ""


def main():
    if not SOURCE.exists():
        print("Error: Not found:", SOURCE)
        return
    DEST.mkdir(exist_ok=True)
    (DEST / "images").mkdir(exist_ok=True)

    for split in SPLITS:
        src_img = SOURCE / split / "images"
        src_lbl = SOURCE / split / "labels"
        if not src_img.exists():
            continue
        dst_img = DEST / "images" / split
        dst_lbl = DEST / "labels" / split
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())
            print("Linked:", dst_img, "->", src_img)
        for txt in src_lbl.glob("*.txt"):
            lines = []
            with open(txt) as f:
                for line in f:
                    c = convert_line(line)
                    if c:
                        lines.append(c)
            (dst_lbl / txt.name).write_text("".join(lines))
        print("Converted labels:", split, "->", dst_lbl)

    yaml = DEST / "weld_2class_big.yaml"
    yaml.write_text(f"""# 2-class from The Welding Defect Dataset: good_weld (0), bad_weld (1)
names:
  - good_weld
  - bad_weld
nc: 2
path: {DEST.resolve()}
train: images/train
val: images/valid
test: images/test
""")
    print("Wrote:", yaml)
    print("\nTrain with: python3 train.py --data data_2class_big/weld_2class_big.yaml --name weld_good_bad")


if __name__ == "__main__":
    main()
