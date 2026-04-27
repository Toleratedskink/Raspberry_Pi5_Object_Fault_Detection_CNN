"""
Sample random labeled images, compare YOLO predictions to ground truth, plot accuracy.

Uses image-level agreement vs GT (see --match-mode). This is **not** the same as mAP from
`model.val()` — a high threshold or missed boxes shows up as **no_detection** and tanks the %.

Curated modes (for slides — read SLIDE_DISCLOSURE.txt):
  python3 random_sample_accuracy.py --curate easy --n 50    # only agreement cases → ~100% on chart
  python3 random_sample_accuracy.py --curate mix --mix-fraction 0.9 --n 40   # ~90% by construction

Unbiased sample (honest error rate on the split):
  python3 random_sample_accuracy.py --curate random --n 50

For official metrics, use model.val() mAP / confusion_matrix.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_dataset_dirs(data_yaml: Path, split: str) -> tuple[Path, Path, list[str]]:
    with open(data_yaml) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    root = Path(cfg["path"]).resolve()
    sub = cfg.get(split) or cfg.get("val" if split == "val" else "test")
    if not sub:
        raise ValueError(f"No split '{split}' in {data_yaml}")
    images_dir = root / sub
    # YOLO layout: images/<split> -> labels/<split>
    rel = Path(sub)
    if rel.parts[0] == "images":
        labels_rel = Path("labels") / Path(*rel.parts[1:])
    else:
        labels_rel = Path("labels") / rel
    labels_dir = root / labels_rel
    if not labels_dir.is_dir():
        labels_dir = root / "labels" / split
    names = list(cfg.get("names") or ["class_0", "class_1"])
    return images_dir, labels_dir, names


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: (4,) xyxy."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    union = ar + br - inter
    return float(inter / union) if union > 0 else 0.0


def gt_boxes_xyxy(label_path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    """YOLO normalized lines -> list of (cls, xyxy numpy)."""
    out: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls not in (0, 1):
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        out.append((cls, np.array([x1, y1, x2, y2], dtype=np.float64)))
    return out


def gt_dominant_from_label(label_path: Path) -> int | None:
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return None
    classes: list[int] = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        c = int(parts[0])
        if c in (0, 1):
            classes.append(c)
    if not classes:
        return None
    return Counter(classes).most_common(1)[0][0]


def find_image(labels_dir: Path, images_dir: Path, stem: str) -> Path | None:
    for ext in EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def collect_pairs(labels_dir: Path, images_dir: Path) -> list[tuple[Path, Path, int]]:
    """List (image_path, label_path, gt_dominant) for valid 2-class labels."""
    pairs: list[tuple[Path, Path, int]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        gd = gt_dominant_from_label(label_path)
        if gd is None:
            continue
        img = find_image(labels_dir, images_dir, label_path.stem)
        if img is None:
            continue
        pairs.append((img, label_path, gd))
    return pairs


def _pred_arrays(r) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns cls (n,), conf (n,), xyxy (n,4) or None."""
    if r.boxes is None or len(r.boxes) == 0:
        return None
    b = r.boxes
    return (
        b.cls.cpu().numpy().astype(int),
        b.conf.cpu().numpy().astype(np.float64),
        b.xyxy.cpu().numpy().astype(np.float64),
    )


def score_single_image(
    model: YOLO,
    img_path: Path,
    lbl_path: Path,
    gt_dom: int,
    conf: float,
    device: str,
    imgsz: int,
    match_mode: str,
    augment: bool,
    iou_thr: float,
) -> tuple[dict, bool, str]:
    """
    One image → row dict, match bool, fail bucket ('ok' | 'no_detection' | 'wrong_class').
    """
    results = model.predict(
        str(img_path),
        conf=conf,
        verbose=False,
        device=device,
        imgsz=imgsz,
        augment=augment,
    )
    r = results[0]
    h, w = r.orig_shape
    pa = _pred_arrays(r)

    pred_dom = -1
    num_boxes = 0
    pred_classes: list[int] = []
    top_conf = 0.0

    if pa is not None:
        clss, confs, _xyxy = pa
        num_boxes = len(clss)
        pred_classes = list(clss)
        top_conf = float(np.max(confs))

        if match_mode == "top_conf":
            scores = np.zeros(max(2, int(clss.max()) + 1), dtype=np.float64)
            for j in range(len(clss)):
                scores[clss[j]] += confs[j]
            pred_dom = int(np.argmax(scores))
        elif match_mode == "dominant":
            pred_dom = Counter(pred_classes).most_common(1)[0][0]
        elif match_mode == "any_class":
            pred_dom = Counter(pred_classes).most_common(1)[0][0]
        elif match_mode == "iou":
            scores = np.zeros(max(2, int(clss.max()) + 1), dtype=np.float64)
            for j in range(len(clss)):
                scores[clss[j]] += confs[j]
            pred_dom = int(np.argmax(scores))
        else:
            pred_dom = Counter(pred_classes).most_common(1)[0][0]

    if num_boxes == 0:
        ok = False
    elif match_mode == "any_class":
        ok = gt_dom in pred_classes
    elif match_mode == "iou":
        gts = gt_boxes_xyxy(lbl_path, w, h)
        dom_gts = [bx for c, bx in gts if c == gt_dom]
        ok = False
        if pa is not None and dom_gts:
            cls_arr, _, xyxy = pa
            for g in dom_gts:
                best_iou = 0.0
                best_j = -1
                for j in range(len(xyxy)):
                    ij = _iou_xyxy(g, xyxy[j])
                    if ij > best_iou:
                        best_iou = ij
                        best_j = j
                if best_j >= 0 and best_iou >= iou_thr and int(cls_arr[best_j]) == gt_dom:
                    ok = True
                    break
    else:
        ok = pred_dom == gt_dom

    if ok:
        bucket = "ok"
    elif num_boxes == 0:
        bucket = "no_detection"
    else:
        bucket = "wrong_class"

    reason = "no_detection" if num_boxes == 0 else ("ok" if ok else "wrong_class")
    row = {
        "image": img_path.name,
        "gt_class": gt_dom,
        "pred_class": pred_dom,
        "top_conf": f"{top_conf:.4f}" if num_boxes else "",
        "num_boxes": num_boxes,
        "match": ok,
        "fail_reason": reason if not ok else "ok",
    }
    return row, ok, bucket


def run_once(
    model: YOLO,
    pairs: list[tuple[Path, Path, int]],
    indices: list[int],
    conf: float,
    device: str,
    imgsz: int,
    match_mode: str,
    augment: bool,
    iou_thr: float,
) -> tuple[int, int, list[dict], dict[str, int]]:
    """Returns correct, total, rows, failure_counts."""
    rows: list[dict] = []
    correct = 0
    total = 0
    fail_counts = {"no_detection": 0, "wrong_class": 0}

    for i in indices:
        img_path, lbl_path, gt_dom = pairs[i]
        row, ok, bucket = score_single_image(
            model,
            img_path,
            lbl_path,
            gt_dom,
            conf,
            device,
            imgsz,
            match_mode,
            augment,
            iou_thr,
        )
        total += 1
        if ok:
            correct += 1
        elif bucket == "no_detection":
            fail_counts["no_detection"] += 1
        else:
            fail_counts["wrong_class"] += 1
        rows.append(row)
    return correct, total, rows, fail_counts


def scan_all_pairs(
    model: YOLO,
    pairs: list[tuple[Path, Path, int]],
    conf: float,
    device: str,
    imgsz: int,
    match_mode: str,
    augment: bool,
    iou_thr: float,
    progress_every: int = 50,
) -> tuple[list[dict], list[int], list[int]]:
    """
    Score every image. Returns rows_by_index[i] = row dict for pairs[i], plus good/bad index lists.
    """
    rows_all: list[dict] = []
    good: list[int] = []
    bad: list[int] = []
    for i, (img_path, lbl_path, gt_dom) in enumerate(pairs):
        if progress_every and i % progress_every == 0 and i > 0:
            print(f"  ... scanned {i}/{len(pairs)}")
        row, ok, _b = score_single_image(
            model, img_path, lbl_path, gt_dom, conf, device, imgsz, match_mode, augment, iou_thr
        )
        rows_all.append(row)
        (good if ok else bad).append(i)
    return rows_all, good, bad


def pick_curated_indices(
    rng: np.random.Generator,
    good: list[int],
    bad: list[int],
    n: int,
    curate: str,
    mix_fraction: float,
) -> tuple[list[int], str]:
    """
    Build index list for the final sample. Returns indices, disclosure one-liner.
    """
    if curate == "easy":
        if len(good) < n:
            print(
                f"Warning: only {len(good)} agreement images; using all ({len(good)} < n={n})."
            )
            return good[:], (
                f"Curated EASY: all {len(good)} images in this split where the model matched labels "
                "(accuracy on this set is 100% by construction)."
            )
        idx = rng.choice(good, size=n, replace=False).tolist()
        return idx, (
            f"Curated EASY: random n={n} images drawn only from cases where the model matched ground truth "
            "(reported accuracy is 100% by construction — use for examples, not unbiased error rate)."
        )

    # mix
    nc = int(round(mix_fraction * n))
    nw = n - nc
    if len(good) < nc or len(bad) < nw:
        print(
            f"Warning: need {nc} correct + {nw} incorrect but have {len(good)} / {len(bad)}. "
            "Taking as many as possible."
        )
        nc = min(nc, len(good))
        nw = min(nw, len(bad))
        if nc + nw == 0:
            return [], "Curated MIX: insufficient images."
    part_g = rng.choice(good, size=nc, replace=False).tolist() if nc else []
    part_b = rng.choice(bad, size=nw, replace=False).tolist() if nw else []
    idx = part_g + part_b
    rng.shuffle(idx)
    acc_pct = 100.0 * nc / (nc + nw) if (nc + nw) else 0.0
    return idx, (
        f"Curated MIX: {nc} images where the model matched labels + {nw} where it did not "
        f"(expected headline accuracy ≈ {acc_pct:.0f}%). "
        "Disclose this in your presentation — it is not a random sample from the split."
    )


def balanced_accuracy_from_rows(rows: list[dict]) -> tuple[float, dict[int, tuple[int, int]]]:
    """Mean of per-class correct/total for gt_class in {0,1}."""
    per: dict[int, list[bool]] = {0: [], 1: []}
    for row in rows:
        g = int(row["gt_class"])
        if g not in per:
            continue
        per[g].append(bool(row["match"]))
    stats: dict[int, tuple[int, int]] = {}
    accs: list[float] = []
    for c in (0, 1):
        ms = per.get(c, [])
        if not ms:
            continue
        t = len(ms)
        k = sum(1 for x in ms if x)
        stats[c] = (k, t)
        accs.append(k / t)
    bal = float(np.mean(accs)) * 100 if accs else 0.0
    return bal, stats


def fail_counts_from_rows(rows: list[dict]) -> dict[str, int]:
    fd = {"no_detection": 0, "wrong_class": 0}
    for r in rows:
        if r["match"]:
            continue
        if r.get("fail_reason") == "no_detection":
            fd["no_detection"] += 1
        else:
            fd["wrong_class"] += 1
    return fd


def plot_results(
    correct: int,
    total: int,
    class_names: list[str],
    out_dir: Path,
    fail_counts: dict[str, int],
    title_note: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wrong = total - correct
    acc = (correct / total * 100) if total else 0.0
    nd = fail_counts.get("no_detection", 0)
    wc = fail_counts.get("wrong_class", 0)
    base_title = f"Image-level agreement ({correct}/{total} = {acc:.1f}%)"
    if title_note:
        base_title = base_title + "\n" + title_note

    # Bar: correct vs wrong
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        ["Correct", "Incorrect"],
        [correct, wrong],
        color=["#27ae60", "#c0392b"],
    )
    ax.set_ylabel("Number of images")
    ax.set_title(base_title)
    for j, v in enumerate([correct, wrong]):
        ax.text(j, v + max(1, total * 0.01), str(int(v)), ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_correct_vs_wrong.png", dpi=150)
    plt.close(fig)

    # Why wrong: no box vs wrong class (explains low headline accuracy)
    if total > 0 and wrong > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(
            ["Correct", "No detection", "Wrong class"],
            [correct, nd, wc],
            color=["#27ae60", "#f39c12", "#c0392b"],
        )
        ax.set_ylabel("Number of images")
        ax.set_title("Breakdown: orange = no boxes above --conf; red = class/Iou mismatch")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_failure_breakdown.png", dpi=150)
        plt.close(fig)

    # Donut
    fig, ax = plt.subplots(figsize=(6, 6))
    if total > 0:
        sizes = [correct, wrong]
        ax.pie(
            sizes,
            labels=[f"Correct\n{correct}", f"Wrong\n{wrong}"],
            autopct="%1.1f%%",
            colors=["#27ae60", "#e74c3c"],
            startangle=90,
            wedgeprops=dict(width=0.45),
        )
    ax.set_title(f"Accuracy: {acc:.1f}%" + (f"\n{title_note}" if title_note else ""))
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_donut.png", dpi=150)
    plt.close(fig)

    # Class name legend file
    legend = (
        "Class IDs (from dataset YAML):\n"
        + "\n".join(f"  {i}: {class_names[i] if i < len(class_names) else '?'}" for i in range(2))
        + "\n\nModes: top_conf = label from highest-score box (default; usually best).\n"
        "iou = GT dominant-class box overlaps a same-class pred with IoU >= --iou-thr.\n"
        "See top_conf in CSV; raise no_detection if --conf is too high.\n"
    )
    (out_dir / "accuracy_notes.txt").write_text(legend)


def main():
    ap = argparse.ArgumentParser(description="Random image sample vs network → accuracy graphs")
    ap.add_argument("--model", default="runs/weld_good_bad22/weights/best.pt", help="Weights .pt")
    ap.add_argument("--data", default="data_2class_big/weld_2class_big.yaml", help="Dataset YAML")
    ap.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="Split to sample from (val matches training validation best)",
    )
    ap.add_argument("--n", type=int, default=50, help="Number of random images per run")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (trials use seed, seed+1, ...)")
    ap.add_argument("--trials", type=int, default=1, help="Repeat with different random draws; >1 plots histogram")
    ap.add_argument(
        "--conf",
        type=float,
        default=0.08,
        help="Min confidence to keep a detection (0.05–0.15 typical; top_conf sums conf per class)",
    )
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size (match training, e.g. 640)")
    ap.add_argument(
        "--augment",
        action="store_true",
        help="Test-time augmentation (multi-scale flips; slower, can help a few percent)",
    )
    ap.add_argument("--device", default="cpu", help="cpu or cuda device id")
    ap.add_argument(
        "--match-mode",
        choices=("top_conf", "dominant", "any_class", "iou"),
        default="top_conf",
        help="top_conf=class with highest sum of confidences; iou=GT box overlap (same class)",
    )
    ap.add_argument(
        "--iou-thr",
        type=float,
        default=0.35,
        help="For match-mode iou: min IoU between a GT box (dominant class) and a same-class pred",
    )
    ap.add_argument(
        "--stratify",
        action="store_true",
        help="Sample equally from each GT class (random mode only)",
    )
    ap.add_argument(
        "--curate",
        choices=("random", "easy", "mix"),
        default="random",
        help="easy=only agreement images (~100 percent on chart); "
        "mix=blend correct+wrong to hit --mix-fraction; random=unbiased sample",
    )
    ap.add_argument(
        "--mix-fraction",
        type=float,
        default=0.9,
        help="With --curate mix: fraction of the n images that are agreement (correct) cases",
    )
    ap.add_argument("--out", default="runs/random_sample_accuracy", help="Output directory")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    model_path = Path(args.model)
    if not data_yaml.exists() or not model_path.exists():
        print("Check --data and --model paths exist.")
        return

    if args.stratify and args.curate != "random":
        print("Note: --stratify applies only to --curate random (ignored here).")

    images_dir, labels_dir, class_names = load_dataset_dirs(data_yaml, args.split)
    pairs = collect_pairs(labels_dir, images_dir)
    if not pairs:
        print(f"No labeled image pairs under {labels_dir} / {images_dir}")
        return

    n = min(args.n, len(pairs))
    if n < args.n:
        print(f"Only {len(pairs)} images available; using n={n}")

    model = YOLO(str(model_path))
    out_root = Path(args.out)
    trial_accs: list[float] = []

    rows_by_index: list[dict] | None = None
    good_idx: list[int] | None = None
    bad_idx: list[int] | None = None
    if args.curate in ("easy", "mix"):
        print(
            f"Scanning all {len(pairs)} labeled images in '{args.split}' (one inference each)..."
        )
        rows_by_index, good_idx, bad_idx = scan_all_pairs(
            model,
            pairs,
            args.conf,
            args.device,
            args.imgsz,
            args.match_mode,
            args.augment,
            args.iou_thr,
        )
        print(f"  Done: {len(good_idx)} agreement / {len(bad_idx)} mismatch.")

    for trial in range(args.trials):
        rng = np.random.default_rng(args.seed + trial)
        disclosure = ""
        title_note = ""
        if args.curate in ("easy", "mix"):
            assert rows_by_index is not None and good_idx is not None and bad_idx is not None
            idx, disclosure = pick_curated_indices(
                rng,
                good_idx,
                bad_idx,
                n,
                args.curate,
                float(args.mix_fraction),
            )
            rows = [rows_by_index[i] for i in idx]
            correct = sum(1 for r in rows if r["match"])
            total = len(rows)
            fail_counts = fail_counts_from_rows(rows)
            if args.curate == "easy":
                title_note = "Curated: agreement cases only (not unbiased error rate)"
            else:
                title_note = (
                    f"Curated mix: ~{100 * args.mix_fraction:.0f}% agreement / "
                    f"~{100 * (1 - args.mix_fraction):.0f}% mismatch by design"
                )
        elif args.stratify:
            by_c: dict[int, list[int]] = {0: [], 1: []}
            for ii, p in enumerate(pairs):
                by_c[p[2]].append(ii)
            half = n // 2
            take: list[int] = []
            for c in (0, 1):
                pool = by_c.get(c, [])
                if not pool:
                    continue
                take.extend(
                    rng.choice(pool, size=min(half, len(pool)), replace=False).tolist()
                )
            rest = n - len(take)
            if rest > 0:
                all_i = set(range(len(pairs)))
                rem = list(all_i - set(take))
                if rem:
                    take.extend(rng.choice(rem, size=min(rest, len(rem)), replace=False).tolist())
            idx = take[:n]
            correct, total, rows, fail_counts = run_once(
                model,
                pairs,
                idx,
                args.conf,
                args.device,
                args.imgsz,
                args.match_mode,
                args.augment,
                args.iou_thr,
            )
        else:
            nn = min(n, len(pairs))
            idx = rng.choice(len(pairs), size=nn, replace=False).tolist()
            correct, total, rows, fail_counts = run_once(
                model,
                pairs,
                idx,
                args.conf,
                args.device,
                args.imgsz,
                args.match_mode,
                args.augment,
                args.iou_thr,
            )
        acc = (correct / total * 100) if total else 0.0
        trial_accs.append(acc)

        sub = out_root if args.trials == 1 else out_root / f"trial_{trial:03d}"
        sub.mkdir(parents=True, exist_ok=True)
        csv_path = sub / "sample_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "gt_class",
                    "pred_class",
                    "top_conf",
                    "num_boxes",
                    "match",
                    "fail_reason",
                ],
            )
            w.writeheader()
            w.writerows(rows)

        if disclosure:
            (sub / "SLIDE_DISCLOSURE.txt").write_text(
                disclosure
                + "\n\nThis subset was chosen to illustrate strong cases or a controlled mix. "
                "Report unbiased performance using model.val() mAP / confusion matrix.\n"
            )
        plot_results(correct, total, class_names, sub, fail_counts, title_note=title_note)
        nd, wc = fail_counts["no_detection"], fail_counts["wrong_class"]
        with_det = total - nd
        acc_det = ((correct / with_det) * 100) if with_det else 0.0
        bal, per_cls = balanced_accuracy_from_rows(rows)
        summ = sub / "accuracy_summary.txt"
        cur_note = ""
        if args.curate == "easy":
            cur_note = "CURATED (easy): agreement-only subset — not an unbiased error rate.\n"
        elif args.curate == "mix":
            cur_note = (
                f"CURATED (mix): ~{100 * args.mix_fraction:.0f}% agreement + "
                f"~{100 * (1 - args.mix_fraction):.0f}% mismatch by design — see SLIDE_DISCLOSURE.txt.\n"
            )
        summ.write_text(
            cur_note
            + f"Overall accuracy: {acc:.2f}% ({correct}/{total})\n"
            + (
                f"Balanced accuracy (mean per-class): {bal:.2f}%\n"
                if args.curate == "random"
                else "Balanced accuracy omitted for curated runs (see per-class split in CSV if needed).\n"
            )
            + f"No detection: {nd}, Wrong class: {wc}\n"
            + "".join(
                f"  Class {k}: {v[0]}/{v[1]} correct\n" for k, v in sorted(per_cls.items())
            )
        )
        print(
            f"Trial {trial + 1}/{args.trials}: {correct}/{total} = {acc:.1f}%  "
            f"(no_detection={nd}, wrong_class={wc})"
        )
        if args.curate == "random" and per_cls:
            c0 = per_cls.get(0)
            c1 = per_cls.get(1)
            c0s = f"{c0[0]}/{c0[1]}" if c0 else "—"
            c1s = f"{c1[0]}/{c1[1]}" if c1 else "—"
            print(f"  Balanced accuracy (mean of per-class rates): {bal:.1f}%  (class0 {c0s}, class1 {c1s})")
        if with_det:
            print(f"  Accuracy on images with ≥1 detection: {acc_det:.1f}% ({with_det} images)")
        print(f"  -> {sub}")

    if args.trials > 1:
        out_root.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(trial_accs, bins=min(15, len(trial_accs)), color="#3498db", edgecolor="white")
        ax.axvline(
            float(np.mean(trial_accs)),
            color="red",
            linestyle="--",
            label=f"mean={np.mean(trial_accs):.1f}%",
        )
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"Accuracy over {args.trials} random samples (n={n} each)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_root / "accuracy_trials_histogram.png", dpi=150)
        plt.close(fig)
        (out_root / "trial_summary.txt").write_text(
            "\n".join(f"trial {i}: {trial_accs[i]:.2f}%" for i in range(len(trial_accs)))
            + f"\n\nmean: {np.mean(trial_accs):.2f}%\nstd: {np.std(trial_accs):.2f}%\n"
        )
        print(f"Saved trial histogram: {out_root / 'accuracy_trials_histogram.png'}")


if __name__ == "__main__":
    main()
