"""
Build a large, slide-ready folder of metrics, tables, and figures for your presentation.

Generates (under --out, default runs/presentation_package):
  - YOLO validation on val + test (Ultralytics plots, predictions JSON, confusion matrices)
  - Matplotlib extras: val vs test comparison, dataset label counts, training curves, confidence histogram
  - Prediction montage (grid of annotated val images)
  - CSV/JSON tables + INDEX.txt listing every file

Usage (from Working/):
  python3 build_presentation_package.py
  python3 build_presentation_package.py --model runs/weld_good_bad22/weights/best.pt --data data_2class_big/weld_2class_big.yaml
  python3 build_presentation_package.py --train-csv runs/weld_good_bad22/results.csv --skip-val   # only charts from existing CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

try:
    import cv2
except ImportError:
    cv2 = None


def _to_jsonable(obj: Any) -> Any:
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


def load_yaml_data_dirs(data_yaml: Path) -> tuple[Path, dict[str, Any]]:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["path"]).resolve(), cfg


def labels_dir_for_split(root: Path, split_cfg_key: str, cfg: dict[str, Any]) -> Path:
    """Resolve labels/... from YAML train|val|test paths (same layout as YOLO)."""
    sub = cfg.get(split_cfg_key)
    if not sub:
        raise ValueError(f"No key '{split_cfg_key}' in dataset YAML")
    rel = Path(sub)
    if rel.parts[0] == "images":
        labels_rel = Path("labels") / Path(*rel.parts[1:])
    else:
        labels_rel = Path("labels") / rel
    return root / labels_rel


def count_boxes_in_labels(labels_dir: Path) -> tuple[int, int, int]:
    """Returns (n_class0, n_class1, n_files_with_content)."""
    n0, n1, nfiles = 0, 0, 0
    if not labels_dir.is_dir():
        return 0, 0, 0
    for p in labels_dir.glob("*.txt"):
        text = p.read_text().strip()
        if not text:
            continue
        nfiles += 1
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            c = int(parts[0])
            if c == 0:
                n0 += 1
            elif c == 1:
                n1 += 1
    return n0, n1, nfiles


def plot_dataset_distribution(
    root: Path,
    cfg: dict[str, Any],
    splits: list[str],
    class_names: list[str],
    out: Path,
) -> None:
    xlabs: list[str] = []
    c0s: list[int] = []
    c1s: list[int] = []
    for sk in splits:
        try:
            ld = labels_dir_for_split(root, sk, cfg)
        except (ValueError, KeyError):
            continue
        n0, n1, _ = count_boxes_in_labels(ld)
        if n0 + n1 == 0:
            continue
        xlabs.append(sk)
        c0s.append(n0)
        c1s.append(n1)

    if not xlabs:
        return

    x = np.arange(len(xlabs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, c0s, width=w, label=class_names[0] if len(class_names) > 0 else "0", color="#3498db")
    ax.bar(x + w / 2, c1s, width=w, label=class_names[1] if len(class_names) > 1 else "1", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabs)
    ax.set_ylabel("Bounding boxes (instances)")
    ax.set_title("Dataset: instances per class by split (from label files)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_val_vs_test(
    metrics_by_split: dict[str, dict[str, float]],
    out: Path,
) -> None:
    splits = list(metrics_by_split.keys())
    if len(splits) < 1:
        return
    keys = [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]
    x = np.arange(len(keys))
    w = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, sp in enumerate(splits):
        rd = metrics_by_split[sp]
        vals = [rd.get(k[0], 0.0) for k in keys]
        offset = (i - len(splits) / 2 + 0.5) * w
        ax.bar(x + offset, vals, width=w * 0.9, label=sp)
    ax.set_xticks(x)
    ax.set_xticklabels([k[1] for k in keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Validation metrics: compare splits")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_radar_style(metrics: dict[str, float], title: str, out: Path) -> None:
    """Simple horizontal bar of the four box metrics (readable on slides)."""
    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    vals = [metrics.get(k, 0.0) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_training_curves(results_csv: Path, out_dir: Path) -> None:
    if not results_csv.exists():
        return
    rows: list[dict[str, str]] = []
    with open(results_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    def col(name: str) -> list[float]:
        return [float(row[name]) for row in rows if name in row and row[name] != ""]

    ax = axes[0, 0]
    if col("train/box_loss"):
        ax.plot(epochs, col("train/box_loss"), label="train box")
    if col("val/box_loss"):
        ax.plot(epochs, col("val/box_loss"), label="val box")
    ax.set_title("Box loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for name, lab in [("train/cls_loss", "train cls"), ("val/cls_loss", "val cls")]:
        if col(name):
            ax.plot(epochs, col(name), label=lab)
    ax.set_title("Classification loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for name, lab in [
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]:
        if col(name):
            ax.plot(epochs, col(name), label=lab)
    ax.set_title("Validation mAP")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for name, lab in [("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall")]:
        if col(name):
            ax.plot(epochs, col(name), label=lab)
    ax.set_title("Precision & recall (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Training history ({results_csv.parent.name})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_training_curves_4panel.png", dpi=150)
    plt.close(fig)

    # Last epoch summary table as image
    last = rows[-1]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    lines = [
        f"Epochs: {last.get('epoch', '')}",
        f"Precision: {float(last.get('metrics/precision(B)', 0)):.4f}   "
        f"Recall: {float(last.get('metrics/recall(B)', 0)):.4f}",
        f"mAP50: {float(last.get('metrics/mAP50(B)', 0)):.4f}   "
        f"mAP50-95: {float(last.get('metrics/mAP50-95(B)', 0)):.4f}",
    ]
    ax.text(0.05, 0.5, "\n".join(lines), fontsize=14, family="monospace", va="center")
    ax.set_title("Final epoch (from results.csv)")
    fig.savefig(out_dir / "fig_training_final_epoch_summary.png", dpi=150)
    plt.close(fig)


def collect_val_image_paths(data_yaml: Path, cfg: dict[str, Any], root: Path, max_n: int) -> list[Path]:
    sub = cfg.get("val") or "images/valid"
    img_dir = root / sub
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    random.shuffle(paths)
    return paths[:max_n]


def plot_confidence_histogram(
    model: YOLO,
    image_paths: list[Path],
    device: str,
    imgsz: int,
    out: Path,
) -> None:
    max_confs: list[float] = []
    for p in image_paths:
        r = model.predict(str(p), conf=0.01, verbose=False, device=device, imgsz=imgsz)[0]
        if r.boxes is None or len(r.boxes) == 0:
            max_confs.append(0.0)
        else:
            max_confs.append(float(r.boxes.conf.max().cpu()))
    if not max_confs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_confs, bins=20, color="#8e44ad", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Max confidence (per image)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of strongest detection confidence (val sample)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def make_montage(
    model: YOLO,
    image_paths: list[Path],
    out: Path,
    device: str,
    imgsz: int,
    conf: float,
) -> None:
    if cv2 is None or len(image_paths) < 1:
        return
    cell_w, cell_h = 320, 240
    paths: list[Path | None] = list(image_paths[:9])
    while len(paths) < 9:
        paths.append(None)
    rows = []
    for r in range(3):
        row_imgs = []
        for c in range(3):
            idx = r * 3 + c
            p = paths[idx]
            if p is None:
                row_imgs.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
            else:
                res = model.predict(
                    str(p),
                    conf=conf,
                    verbose=False,
                    device=device,
                    imgsz=imgsz,
                )[0]
                im = res.plot()
                row_imgs.append(cv2.resize(im, (cell_w, cell_h)))
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)
    cv2.imwrite(str(out), grid)


def write_dataset_csv(root: Path, cfg: dict[str, Any], splits: list[str], out: Path, class_names: list[str]) -> None:
    rows_out = []
    for sk in splits:
        try:
            ld = labels_dir_for_split(root, sk, cfg)
        except (ValueError, KeyError):
            continue
        n0, n1, nf = count_boxes_in_labels(ld)
        rows_out.append(
            {
                "split": sk,
                "label_files_nonempty": nf,
                "instances_class0": n0,
                "instances_class1": n1,
                "class0_name": class_names[0] if len(class_names) > 0 else "0",
                "class1_name": class_names[1] if len(class_names) > 1 else "1",
            }
        )
    with open(out, "w", newline="") as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)


def main():
    ap = argparse.ArgumentParser(description="Build a large presentation package (metrics + graphs)")
    ap.add_argument("--model", default="runs/weld_good_bad22/weights/best.pt")
    ap.add_argument("--data", default="data_2class_big/weld_2class_big.yaml")
    ap.add_argument("--out", default="runs/presentation_package", help="Output folder")
    ap.add_argument("--train-csv", default="runs/weld_good_bad22/results.csv", help="Training results.csv for curves")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--montage-conf", type=float, default=0.25)
    ap.add_argument("--skip-val", action="store_true", help="Skip YOLO val (only dataset + training plots)")
    ap.add_argument("--val-splits", default="val,test", help="Comma-separated splits for model.val()")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    tables = out / "tables"
    yolo_runs = out / "yolo_validation"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    yolo_runs.mkdir(exist_ok=True)

    data_yaml = Path(args.data)
    model_path = Path(args.model)
    root, cfg = load_yaml_data_dirs(data_yaml)
    class_names = list(cfg.get("names") or ["bad_weld", "good_weld"])

    index_lines: list[str] = [f"Presentation package: {out.resolve()}\n", "=" * 60 + "\n"]

    # --- Dataset tables + figure ---
    write_dataset_csv(root, cfg, ["train", "val", "test"], tables / "dataset_instances_by_split.csv", class_names)
    index_lines.append("tables/dataset_instances_by_split.csv\n")
    plot_dataset_distribution(
        root, cfg, ["train", "val", "test"], class_names, figures / "fig_dataset_instances_by_split.png"
    )
    index_lines.append("figures/fig_dataset_instances_by_split.png\n")

    # --- Training curves ---
    train_csv = Path(args.train_csv)
    plot_training_curves(train_csv, figures)
    if train_csv.exists():
        index_lines.append("figures/fig_training_curves_4panel.png\n")
        index_lines.append("figures/fig_training_final_epoch_summary.png\n")

    if args.skip_val or not model_path.exists():
        (out / "INDEX.txt").write_text("".join(index_lines))
        print(f"Wrote partial package to {out} (skipped validation)")
        return

    model = YOLO(str(model_path))
    splits = [s.strip() for s in args.val_splits.split(",") if s.strip()]
    metrics_by_split: dict[str, dict[str, float]] = {}
    summaries: dict[str, Any] = {}

    for sp in splits:
        name = f"yolo_{sp}"
        print(f"Running validation split={sp} ...")
        metrics = model.val(
            data=str(data_yaml),
            split=sp,
            imgsz=args.imgsz,
            batch=8,
            plots=True,
            save_json=True,
            project=str(yolo_runs),
            name=name,
            device=args.device,
            verbose=False,
        )
        rd = dict(metrics.results_dict)
        metrics_by_split[sp] = rd
        summaries[sp] = {
            "results_dict": _to_jsonable(rd),
            "per_class": _to_jsonable(metrics.summary()),
        }
        plot_radar_style(
            rd,
            f"Box metrics ({sp} split)",
            figures / f"fig_metrics_horizontal_{sp}.png",
        )
        index_lines.append(f"yolo_validation/{name}/ (Ultralytics plots, confusion_matrix, PR curves)\n")
        index_lines.append(f"figures/fig_metrics_horizontal_{sp}.png\n")

    with open(tables / "metrics_by_split.json", "w") as f:
        json.dump(_to_jsonable(summaries), f, indent=2)

    # Comparison chart
    if len(metrics_by_split) >= 1:
        plot_val_vs_test(metrics_by_split, figures / "fig_metrics_compare_splits.png")
        index_lines.append("figures/fig_metrics_compare_splits.png\n")
        index_lines.append("tables/metrics_by_split.json\n")

    # CSV row for quick paste
    with open(tables / "metrics_compare_splits.csv", "w", newline="") as f:
        fieldnames = ["split", "precision", "recall", "mAP50", "mAP50-95"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sp, rd in metrics_by_split.items():
            w.writerow(
                {
                    "split": sp,
                    "precision": rd.get("metrics/precision(B)", 0),
                    "recall": rd.get("metrics/recall(B)", 0),
                    "mAP50": rd.get("metrics/mAP50(B)", 0),
                    "mAP50-95": rd.get("metrics/mAP50-95(B)", 0),
                }
            )

    # Confidence histogram + montage (val images)
    val_paths = collect_val_image_paths(data_yaml, cfg, root, max_n=120)
    if val_paths:
        plot_confidence_histogram(
            model,
            val_paths[:80],
            args.device,
            args.imgsz,
            figures / "fig_confidence_histogram_val.png",
        )
        index_lines.append("figures/fig_confidence_histogram_val.png\n")
        random.seed(42)
        random.shuffle(val_paths)
        make_montage(
            model,
            val_paths[:9],
            figures / "montage_val_predictions_3x3.jpg",
            args.device,
            args.imgsz,
            args.montage_conf,
        )
        index_lines.append("figures/montage_val_predictions_3x3.jpg\n")

    index_lines.append("\nKey Ultralytics PNGs (under yolo_validation/<run>/):\n")
    for sub in sorted(yolo_runs.iterdir()):
        if not sub.is_dir():
            continue
        for pat in (
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxPR_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "BoxF1_curve.png",
        ):
            p = sub / pat
            if p.exists():
                index_lines.append(f"  {p.relative_to(out)}\n")

    index_lines.append("\nSuggested slide order:\n")
    index_lines.append("  1. fig_dataset_instances_by_split.png\n")
    index_lines.append("  2. fig_training_curves_4panel.png\n")
    index_lines.append("  3. confusion_matrix + BoxPR from yolo_validation\n")
    index_lines.append("  4. fig_metrics_compare_splits.png + fig_metrics_horizontal_*.png\n")
    index_lines.append("  5. fig_confidence_histogram_val.png\n")
    index_lines.append("  6. montage_val_predictions_3x3.jpg\n")
    (out / "INDEX.txt").write_text("".join(index_lines))

    print(f"Done. Open: {out / 'INDEX.txt'}")


if __name__ == "__main__":
    main()
