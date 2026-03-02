"""
WoundAI — Visual Dataset Inspector (FIXED)
===========================================
Correctly handles your folder structure:

    MAD/
    ├── OOD/
    │   ├── class_a/  (images here)
    │   └── class_b/
    ├── Wound_Detection/
    │   ├── wound/
    │   └── non_wound/
    └── Wound_Type/
        ├── burn/
        ├── laceration/
        └── ...

Usage (from inside MAD/ folder):
    python Inspect_Dataset.py
    python Inspect_Dataset.py --mode all          ← inspect all 3 folders
    python Inspect_Dataset.py --mode wound_type   ← inspect Wound_Type only
    python Inspect_Dataset.py --mode detection    ← inspect Wound_Detection only
    python Inspect_Dataset.py --mode ood          ← inspect OOD only
"""

import os
import json
import random
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─── TIMELINE TRACKER ─────────────────────────────────────────────────────────
class Timeline:
    """Tracks start/end time and completion status for every pipeline step."""
    _ICON = {"completed": "✅", "failed": "❌", "skipped": "⏭ ", "in_progress": "🔄"}

    def __init__(self, name: str):
        self.name   = name
        self.steps  = {}
        self._start = datetime.now()

    def start(self, step: str):
        self.steps[step] = {
            "status": "in_progress",
            "started": datetime.now().isoformat(),
            "finished": None, "duration_s": None, "note": ""
        }
        print(f"  ⏱  [{datetime.now().strftime('%H:%M:%S')}] START   {step}")

    def done(self, step: str, note: str = ""):
        self._close(step, "completed", note)

    def fail(self, step: str, reason: str = ""):
        self._close(step, "failed", reason)

    def skipped(self, step: str, reason: str = ""):
        self.steps[step] = {
            "status": "skipped", "started": None,
            "finished": None, "duration_s": None, "note": reason
        }
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"  ⏭  [{ts}] SKIP    {step}" + (f"  — {reason}" if reason else ""))

    def _close(self, step: str, status: str, note: str):
        if step not in self.steps:
            self.steps[step] = {"started": datetime.now().isoformat()}
        now = datetime.now()
        dur = round((now - datetime.fromisoformat(
            self.steps[step]["started"])).total_seconds(), 1)
        self.steps[step].update({
            "status": status, "finished": now.isoformat(),
            "duration_s": dur, "note": note
        })
        icon  = self._ICON.get(status, "•")
        label = "DONE  " if status == "completed" else "FAILED"
        ts    = now.strftime('%H:%M:%S')
        print(f"  {icon} [{ts}] {label}  {step}  ({dur}s)" +
              (f"  — {note}" if note else ""))

    def print_summary(self):
        total = round((datetime.now() - self._start).total_seconds(), 1)
        print(f"\n{'─'*60}")
        print(f"  TIMELINE SUMMARY  —  {self.name}")
        print(f"{'─'*60}")
        for step, info in self.steps.items():
            icon = self._ICON.get(info.get("status", ""), "•")
            dur  = f"  ({info['duration_s']}s)" if info.get("duration_s") else ""
            note = f"  — {info['note']}" if info.get("note") else ""
            print(f"  {icon} {step}{dur}{note}")
        print(f"{'─'*60}")
        print(f"  Total elapsed: {total}s  ({total/60:.1f} min)")
        print(f"{'─'*60}\n")

    def save(self, path: Path):
        total = round((datetime.now() - self._start).total_seconds(), 1)
        data  = {
            "pipeline": self.name,
            "started_at": self._start.isoformat(),
            "saved_at": datetime.now().isoformat(),
            "total_elapsed_s": total,
            "steps": self.steps
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  📋 Timeline saved → {path}")


# ─── FOLDER DETECTION ─────────────────────────────────────────────────────────
def find_project_root() -> Path:
    """Auto-detect the MAD project root by looking for known subfolders."""
    cwd = Path.cwd()
    # If we're already inside MAD/ and can see the 3 folders
    for candidate in [cwd, cwd.parent]:
        has_wd = (candidate / "Wound_Detection").exists()
        has_wt = (candidate / "Wound_Type").exists()
        has_ood = (candidate / "OOD").exists()
        if has_wd or has_wt or has_ood:
            return candidate
    return cwd


def discover_datasets(root: Path) -> dict:
    """
    Find all dataset folders and their class subfolders.
    Reads from raw source folders (root/Wound_Detection, root/Wound_Type, root/OOD)
    to document the original dataset before preprocessing.
    Returns: {dataset_name: {class_name: [image_paths]}}
    """
    known = {
        "Wound_Detection": root / "Wound_Detection",
        "Wound_Type":      root / "Wound_Type",
        "OOD":             root / "OOD",
    }

    discovered = {}
    for ds_name, ds_path in known.items():
        if not ds_path.exists():
            print(f"  ⚠️  Not found: {ds_path}")
            continue

        classes = {}
        # Check if images are directly in ds_path (flat structure)
        direct_imgs = [f for f in ds_path.iterdir()
                       if f.is_file() and f.suffix.lower() in IMG_EXTS]

        if direct_imgs:
            # Flat: all images directly in the folder → treat as single class
            classes[ds_name] = direct_imgs
        else:
            # Nested: subfolders = classes
            for cls_dir in sorted(ds_path.iterdir()):
                if not cls_dir.is_dir():
                    continue
                imgs = [f for f in cls_dir.rglob("*")
                        if f.is_file() and f.suffix.lower() in IMG_EXTS]
                if imgs:
                    classes[cls_dir.name] = imgs

        if classes:
            discovered[ds_name] = classes
            total = sum(len(v) for v in classes.values())
            print(f"  ✅ {ds_name:<20} {len(classes)} classes, {total} images")
        else:
            print(f"  ⚠️  {ds_name:<20} folder exists but no images found inside")

    return discovered


# ─── CLASS GRID ───────────────────────────────────────────────────────────────
def draw_class_grid(classes: dict, title: str, output_path: Path,
                    n_cols: int = 8, cell: int = 120):
    """One row per class, n_cols sample images."""
    if not classes:
        print(f"  ⚠️  No classes to draw for: {title}")
        return

    class_names = list(classes.keys())
    n_rows = len(class_names)

    fig_w = n_cols * (cell / 72) + 2.5
    fig_h = n_rows * (cell / 72) + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0a0f1e")
    fig.suptitle(title, color="white", fontsize=11, y=0.99)

    for row_i, cls_name in enumerate(class_names):
        img_paths = classes[cls_name]
        samples   = random.sample(img_paths, min(n_cols, len(img_paths)))
        n_total   = len(img_paths)

        for col_i in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, row_i * n_cols + col_i + 1)

            if col_i < len(samples):
                img = cv2.imread(str(samples[col_i]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (cell, cell))
                    ax.imshow(img)

            ax.axis("off")

            if col_i == 0:
                label = cls_name.replace("_", "\n")
                ax.text(-0.12, 0.5, f"{label}\n({n_total})",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=6, color="white", fontweight="bold")

    plt.subplots_adjust(wspace=0.02, hspace=0.06,
                        left=0.14, right=0.99, top=0.97, bottom=0.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=130, facecolor="#0a0f1e",
                bbox_inches="tight")
    plt.close()
    print(f"  ✅ Class grid → {output_path}")


# ─── DISTRIBUTION CHART ───────────────────────────────────────────────────────
def draw_distribution(classes: dict, title: str, output_path: Path):
    """Horizontal bar chart coloured by imbalance severity."""
    if not classes:
        return

    names  = list(classes.keys())
    counts = [len(classes[c]) for c in names]
    max_c  = max(counts) if counts else 1

    colours = []
    for c in counts:
        r = max_c / max(c, 1)
        colours.append(
            "#ef4444" if r > 5 else
            "#f59e0b" if r > 2 else
            "#10b981"
        )

    fig, ax = plt.subplots(
        figsize=(10, max(3, len(names) * 0.5)),
        facecolor="#0a0f1e"
    )
    ax.set_facecolor("#0f172a")

    bars = ax.barh(names, counts, color=colours, edgecolor="none", height=0.6)
    ax.set_xlabel("Number of Images", color="white")
    ax.set_title(f"{title}\n🔴 Severely imbalanced  🟡 Imbalanced  🟢 OK",
                 color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    for i, (c, _) in enumerate(zip(counts, names)):
        ax.text(c + max_c * 0.01, i, str(c),
                va="center", fontsize=8, color="white")

    if max_c > 1:
        ax.axvline(max_c * 0.5, color="#f59e0b", lw=1,
                   linestyle="--", alpha=0.7, label="50% of max")
        ax.axvline(max_c * 0.2, color="#ef4444", lw=1,
                   linestyle="--", alpha=0.7, label="20% of max")
        ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=130, facecolor="#0a0f1e",
                bbox_inches="tight")
    plt.close()
    print(f"  ✅ Distribution chart → {output_path}")


# ─── IMAGE STATS ──────────────────────────────────────────────────────────────
def analyse_stats(classes: dict, n_sample: int = 150) -> dict:
    """Compute brightness / contrast / saturation on a random sample."""
    all_imgs = [p for imgs in classes.values() for p in imgs]
    if not all_imgs:
        return {}

    sample = random.sample(all_imgs, min(n_sample, len(all_imgs)))
    brightness, contrast, saturation = [], [], []

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness.append(float(gray.mean()))
        contrast.append(float(gray.std()))
        saturation.append(float(hsv[:, :, 1].mean()))

    if not brightness:
        return {}

    return {
        "n_sampled":  len(brightness),
        "brightness": {"mean": float(np.mean(brightness)),
                       "std":  float(np.std(brightness))},
        "contrast":   {"mean": float(np.mean(contrast)),
                       "std":  float(np.std(contrast))},
        "saturation": {"mean": float(np.mean(saturation)),
                       "std":  float(np.std(saturation))},
    }


# ─── LATEX TABLE ──────────────────────────────────────────────────────────────
def make_latex_table(all_datasets: dict, output_path: Path):
    lines = [
        r"% Dataset Summary Table — paste into your paper",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Dataset Summary}",
        r"\label{tab:dataset}",
        r"\begin{tabular}{llr}",
        r"\hline",
        r"\textbf{Dataset} & \textbf{Class} & \textbf{Images} \\",
        r"\hline",
    ]
    grand_total = 0
    for ds_name, classes in all_datasets.items():
        ds_total = sum(len(v) for v in classes.values())
        for i, (cls_name, imgs) in enumerate(sorted(classes.items())):
            ds_label = ds_name.replace("_", " ") if i == 0 else ""
            lines.append(
                f"  {ds_label:<20} & "
                f"{cls_name.replace('_',' ').title():<25} & "
                f"{len(imgs):>5} \\\\"
            )
        lines.append(
            f"  \\multicolumn{{2}}{{r}}{{\\textit{{subtotal}}}} & "
            f"\\textit{{{ds_total}}} \\\\"
        )
        lines.append(r"  \hline")
        grand_total += ds_total

    lines += [
        f"  \\textbf{{Total}} & & \\textbf{{{grand_total}}} \\\\",
        r"  \hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✅ LaTeX table → {output_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",       default=None,
                        help="Path to MAD/ project root (auto-detected if not given)")
    parser.add_argument("--output_dir", default="./inspection")
    parser.add_argument("--mode",       default="all",
                        choices=["all", "wound_type", "detection", "ood"])
    parser.add_argument("--n_cols",     type=int, default=8,
                        help="Images per row in class grid")
    args = parser.parse_args()

    root    = Path(args.root) if args.root else find_project_root()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  WoundAI — Dataset Inspector")
    print("=" * 55)
    print(f"  Project root : {root}")
    print(f"  Output       : {out_dir}")
    print(f"  Mode         : {args.mode}")
    print(f"  Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    timeline = Timeline("WoundAI Dataset Inspection")

    # ── Discover all datasets ─────────────────────────────────────────────────
    timeline.start("Discover datasets")
    all_datasets = discover_datasets(root)

    if not all_datasets:
        timeline.fail("Discover datasets", "no datasets found")
        print("\n❌  No datasets found!")
        print(f"    Looked in: {root}")
        print("    Expected subfolders: Wound_Detection/, Wound_Type/, OOD/")
        print("    Each containing class subfolders with images.\n")
        print("    Try: python Inspect_Dataset.py --root /full/path/to/MAD")
        return
    timeline.done("Discover datasets",
                  f"{len(all_datasets)} dataset(s) found")

    # ── Filter by mode ────────────────────────────────────────────────────────
    mode_map = {
        "wound_type": ["Wound_Type"],
        "detection":  ["Wound_Detection"],
        "ood":        ["OOD"],
        "all":        list(all_datasets.keys()),
    }
    selected = {k: v for k, v in all_datasets.items()
                if k in mode_map[args.mode]}

    # ── Print counts ──────────────────────────────────────────────────────────
    print()
    for ds_name, classes in selected.items():
        total = sum(len(v) for v in classes.values())
        print(f"  📁 {ds_name}  ({len(classes)} classes, {total} images)")
        for cls_name, imgs in sorted(classes.items(),
                                     key=lambda x: -len(x[1])):
            bar = "█" * min(40, len(imgs) // 2)
            print(f"     {cls_name:<28} {len(imgs):>5}  {bar}")

    # ── Generate outputs ──────────────────────────────────────────────────────
    print("\n  Generating visuals...")
    for ds_name, classes in selected.items():
        timeline.start(f"Inspect {ds_name}")
        ds_out = out_dir / ds_name

        # Class grid
        draw_class_grid(
            classes,
            title=f"{ds_name} — Sample Images per Class",
            output_path=ds_out / "class_grid.jpg",
            n_cols=args.n_cols
        )

        # Distribution chart
        draw_distribution(
            classes,
            title=f"{ds_name} — Class Distribution",
            output_path=ds_out / "class_distribution.png"
        )

        # Image stats
        print(f"\n  📊 Image statistics for {ds_name}:")
        stats = analyse_stats(classes)
        if stats:
            b = stats["brightness"]
            c = stats["contrast"]
            s = stats["saturation"]
            print(f"     Brightness : {b['mean']:.1f} ± {b['std']:.1f}"
                  f"  (good range: 60–200)")
            print(f"     Contrast   : {c['mean']:.1f} ± {c['std']:.1f}"
                  f"  (want: > 30)")
            print(f"     Saturation : {s['mean']:.1f} ± {s['std']:.1f}")

            # Warnings
            if b["mean"] < 60:
                print("     ⚠️  Images are too DARK on average — "
                      "check your preprocessing")
            elif b["mean"] > 200:
                print("     ⚠️  Images are OVEREXPOSED on average")
            else:
                print("     ✅  Brightness looks good")

            if c["mean"] < 30:
                print("     ⚠️  Low contrast — CLAHE may not have run correctly")
            else:
                print("     ✅  Contrast looks good")

            # Save stats
            with open(ds_out / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)
        else:
            print("     ⚠️  Could not read any images for stats")

        n_imgs = sum(len(v) for v in classes.values())
        timeline.done(f"Inspect {ds_name}",
                      f"{len(classes)} classes, {n_imgs} images")

    # ── LaTeX table (all datasets) ───────────────────────────────────────────────
    timeline.start("Generate LaTeX table")
    make_latex_table(selected, out_dir / "dataset_table.tex")
    timeline.done("Generate LaTeX table")

    # ── Final summary ─────────────────────────────────────────────────────────
    grand_total = sum(
        len(imgs)
        for classes in selected.values()
        for imgs in classes.values()
    )
    print(f"\n{'='*55}")
    print(f"  TOTAL IMAGES ACROSS ALL DATASETS: {grand_total}")
    print(f"{'='*55}")
    print(f"\n  ✅ All outputs saved to: {out_dir}")
    print(f"\n  📂 Open these to review:")
    for ds_name in selected:
        print(f"     {out_dir}/{ds_name}/class_grid.jpg")
        print(f"     {out_dir}/{ds_name}/class_distribution.png")
    print(f"     {out_dir}/dataset_table.tex ")

    timeline.save(out_dir / "timeline.json")
    timeline.print_summary()


if __name__ == "__main__":
    main()
