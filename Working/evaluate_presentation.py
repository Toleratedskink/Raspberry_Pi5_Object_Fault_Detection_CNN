"""
Evaluate the trained detector on a test/val split and save metrics + graphs for presentations.

- Runs Ultralytics validation (mAP, precision, recall) and saves PR-style plots from YOLO.
- Saves matplotlib bar charts: overall metrics and per-class metrics.
- Optional: image-level "agreement rate" using only converted labels in data_2class_big/labels
  (reads labels from this folder so symlink issues with the big dataset are avoided).

Usage (from Working/):
  source venv/bin/activate
  python3 evaluate_presentation.py
  python3 evaluate_presentation.py --split test --model runs/weld_good_bad2/weights/best.pt
  python3 evaluate_presentation.py --image-level
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def _to_jsonable(obj):
    """Convert numpy scalars and nested structures to JSON-serializable Python types."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item") and callable(getattr(obj, "item")) and not isinstance(obj, (dict, list, tuple)):
        try:
            return _to_jsonable(obj.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    return obj


def _plot_metrics_bars(
    names: list[str],
    precision: list[float],
    recall: list[float],
    map50: list[float],
    out_path: Path,
    title: str,
) -> None:
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, precision, width=w, label="Precision", color="#2ecc71")
    ax.bar(x, recall, width=w, label="Recall", color="#3498db")
    ax.bar(x + w, map50, width=w, label="mAP50", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_overall_bars(metrics_dict: dict[str, float], out_path: Path) -> None:
    keys = ["Precision", "Recall", "mAP50", "mAP50-95"]
    vals = [
        metrics_dict.get("metrics/precision(B)", 0),
        metrics_dict.get("metrics/recall(B)", 0),
        metrics_dict.get("metrics/mAP50(B)", 0),
        metrics_dict.get("metrics/mAP50-95(B)", 0),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
    ax.bar(keys, vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall detection metrics (test/val split)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_val_and_plots(
    model_path: Path,
    data_yaml: Path,
    split: str,
    out_dir: Path,
    imgsz: int,
    device: str,
) -> object:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=8,
        plots=True,
        save_json=True,
        project=str(out_dir),
        name="yolo_val",
        device=device,
        verbose=False,
    )
    return metrics


def image_level_agreement(
    model_path: Path,
    labels_dir: Path,
    images_dir: Path,
    conf: float,
    device: str,
) -> tuple[int, int, list[dict]]:
    """
    For each label file with only classes 0/1, compare dominant GT class vs dominant prediction.
    Returns correct count, total count, rows for CSV.
    """
    model = YOLO(str(model_path))
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    correct = 0
    total = 0
    rows: list[dict] = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = label_path.read_text().strip().splitlines()
        if not lines:
            continue
        gt_classes = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            c = int(parts[0])
            if c not in (0, 1):
                continue  # skip incompatible labels for 2-class eval
            gt_classes.append(c)
        if not gt_classes:
            continue

        stem = label_path.stem
        img_path = None
        for ext in exts:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        # Dominant GT: majority vote (or max count of bad_weld vs good_weld)
        gt_dom = int(np.round(np.mean(gt_classes)))  # simple; or mode
        from collections import Counter

        gt_dom = Counter(gt_classes).most_common(1)[0][0]

        results = model.predict(str(img_path), conf=conf, verbose=False, device=device)
        r = results[0]
        pred_classes = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                pred_classes.append(int(box.cls[0]))
        if pred_classes:
            pred_dom = Counter(pred_classes).most_common(1)[0][0]
        else:
            pred_dom = -1  # no detection

        ok = pred_dom == gt_dom
        if pred_dom != -1:
            total += 1
            if ok:
                correct += 1

        rows.append(
            {
                "image": img_path.name,
                "gt_dominant": gt_dom,
                "pred_dominant": pred_dom,
                "match": ok if pred_dom != -1 else "",
            }
        )

    return correct, total, rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate model and save presentation graphs.")
    ap.add_argument("--model", default="runs/weld_good_bad2/weights/best.pt", help="Path to .pt weights")
    ap.add_argument("--data", default="data_2class_big/weld_2class_big.yaml", help="Dataset YAML")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"), help="Split to evaluate")
    ap.add_argument("--out", default="runs/presentation_eval", help="Output directory")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size for validation")
    ap.add_argument("--device", default="cpu", help="cpu or 0 / cuda")
    ap.add_argument(
        "--image-level",
        action="store_true",
        help="Also compute image-level agreement using data_2class_big labels (avoids symlink label bugs)",
    )
    ap.add_argument("--conf", type=float, default=0.25, help="Conf threshold for image-level step")
    args = ap.parse_args()

    model_path = Path(args.model)
    data_yaml = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    if not data_yaml.exists():
        print(f"Dataset YAML not found: {data_yaml}")
        return

    print("Running Ultralytics validation (this may take a few minutes)...")
    metrics = run_val_and_plots(model_path, data_yaml, args.split, out_dir, args.imgsz, args.device)

    rd = metrics.results_dict
    summary = metrics.summary()

    # Save JSON summary
    summary_path = out_dir / "metrics_summary.json"
    payload = {
        "split": args.split,
        "results_dict": _to_jsonable(dict(rd)),
        "per_class": _to_jsonable(summary),
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {summary_path}")

    # CSV per-class
    csv_path = out_dir / "per_class_metrics.csv"
    if summary:
        keys = summary[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in summary:
                w.writerow({k: (float(v) if hasattr(v, "item") else v) for k, v in row.items()})
        print(f"Saved: {csv_path}")

    # Matplotlib figures
    names = [str(row.get("Class", f"class_{i}")) for i, row in enumerate(summary)]
    prec = [float(row["Box-P"]) for row in summary]
    rec = [float(row["Box-R"]) for row in summary]
    m50 = [float(row["mAP50"]) for row in summary]
    _plot_metrics_bars(
        names,
        prec,
        rec,
        m50,
        out_dir / "fig_per_class_metrics.png",
        f"Per-class metrics ({args.split} split)",
    )
    _plot_overall_bars(rd, out_dir / "fig_overall_metrics.png")
    print(f"Saved: {out_dir / 'fig_per_class_metrics.png'}")
    print(f"Saved: {out_dir / 'fig_overall_metrics.png'}")

    # "Success rate" slide-friendly number (mAP50 as overall detection quality)
    success = rd.get("metrics/mAP50(B)", 0) * 100
    print(f"\nOverall mAP50 (detection quality): {success:.1f}%")
    (out_dir / "README_presentation.txt").write_text(
        f"Split: {args.split}\n"
        f"Overall Precision: {rd.get('metrics/precision(B)', 0):.4f}\n"
        f"Overall Recall: {rd.get('metrics/recall(B)', 0):.4f}\n"
        f"Overall mAP50: {rd.get('metrics/mAP50(B)', 0):.4f}\n"
        f"Overall mAP50-95: {rd.get('metrics/mAP50-95(B)', 0):.4f}\n"
        f"\nYOLO also saved plots under: {out_dir / 'yolo_val'}\n"
        f"Use fig_overall_metrics.png and fig_per_class_metrics.png in slides.\n"
    )
    print(f"Saved: {out_dir / 'README_presentation.txt'}")

    if args.image_level:
        labels_dir = Path("data_2class_big/labels/test")
        images_dir = Path("data_2class_big/images/test")
        if labels_dir.is_dir() and images_dir.is_dir():
            c, t, rows = image_level_agreement(model_path, labels_dir, images_dir, args.conf, args.device)
            rate = (c / t * 100) if t else 0.0
            p = out_dir / "image_level_agreement.csv"
            with open(p, "w", newline="") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=rows[0].keys())
                    w.writeheader()
                    w.writerows(rows)
            print(f"\nImage-level agreement (dominant class vs GT): {c}/{t} = {rate:.1f}%")
            print(f"Saved: {p}")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Match rate"], [rate], color="#27ae60")
            ax.set_ylim(0, 105)
            ax.set_ylabel("Percent")
            ax.set_title("Image-level class agreement (test, valid labels only)")
            fig.tight_layout()
            fig.savefig(out_dir / "fig_image_level_agreement.png", dpi=150)
            plt.close(fig)
            print(f"Saved: {out_dir / 'fig_image_level_agreement.png'}")
        else:
            print("Skipping image-level: data_2class_big test folders not found.")


if __name__ == "__main__":
    main()
