"""
WoundAI — Prepare Datasets
============================
Reads your existing folder structure:

    MAD/
    ├── datasets_clean/        ← preprocessed images (from Preprocess.py)
    │   ├── Wound_Detection/
    │   │   ├── wound/
    │   │   └── non_wound/
    │   ├── Wound_Type/
    │   │   ├── Bruise/
    │   │   ├── Burn/
    │   │   └── ...
    │   └── OOD/
    │       ├── Miscellenous_wounds/
    │       ├── Orthopedic_wounds/
    │       └── Abdominal_wounds/

Produces:

    MAD/
    └── datasets/
        ├── dataset1_binary/       ← Stage 1 training (wound vs non_wound)
        │   ├── wound/
        │   └── non_wound/
        ├── dataset2_multiclass/   ← Stage 2 training (wound types)
        │   ├── Bruise/
        │   ├── Burn/
        │   └── ...
        └── dataset3_ood/          ← Stage 3 evaluation only
            └── ood/

Usage (run from MAD/ folder):
    python prepare_datasets.py
    python prepare_datasets.py --clean_dir ./datasets_clean --out_dir ./datasets
"""

import shutil
import json
import argparse
from pathlib import Path
from collections import defaultdict
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


def get_images(folder: Path) -> list[Path]:
    return [f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in IMG_EXTS]


def copy_images(src_folder: Path, dst_folder: Path) -> int:
    """Copy all images from src to dst. Returns count copied."""
    dst_folder.mkdir(parents=True, exist_ok=True)
    imgs = get_images(src_folder)
    for img in imgs:
        dst = dst_folder / img.name
        # Avoid name collisions
        if dst.exists():
            dst = dst_folder / f"{img.stem}_{img.parent.name}{img.suffix}"
        shutil.copy2(img, dst)
    return len(imgs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", default="./datasets_clean",
                        help="Output of Preprocess.py (default: ./datasets_clean)")
    parser.add_argument("--out_dir",   default="./datasets",
                        help="Where to write training datasets (default: ./datasets)")
    args = parser.parse_args()

    clean  = Path(args.clean_dir)
    out    = Path(args.out_dir)

    # Destination folders
    ds1 = out / "dataset1_binary"
    ds2 = out / "dataset2_multiclass"
    ds3 = out / "dataset3_ood"

    print("=" * 60)
    print("  WoundAI — Prepare Datasets")
    print("=" * 60)
    print(f"  Source : {clean}")
    print(f"  Output : {out}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    timeline = Timeline("WoundAI Prepare Datasets")

    # Validate source exists
    if not clean.exists():
        timeline.fail("Validate source", str(clean))
        print(f"❌ Source folder not found: {clean}")
        print("   Run Preprocess.py first, or check the --clean_dir path.")
        return
    timeline.done("Validate source", str(clean))

    stats = defaultdict(int)

    # ── DATASET 1: Binary (wound vs non_wound) ────────────────────────────────    timeline.start("Dataset 1 — Binary (wound / non_wound)")    print("📁 Building Dataset 1 — Binary (wound / non_wound)")
    wd_folder = clean / "Wound_Detection"

    if wd_folder.exists():
        wound_dir    = wd_folder / "wound"
        nonwound_dir = wd_folder / "non_wound"

        if wound_dir.exists():
            n = copy_images(wound_dir, ds1 / "wound")
            print(f"   wound     : {n} images")
            stats["ds1_wound"] = n
        else:
            # Fallback: if no subfolders, treat all images as wound
            imgs = get_images(wd_folder)
            if imgs:
                (ds1 / "wound").mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    shutil.copy2(img, ds1 / "wound" / img.name)
                print(f"   wound     : {len(imgs)} images  "
                      f"(⚠️  no wound/ subfolder found — used all images)")
                stats["ds1_wound"] = len(imgs)

        if nonwound_dir.exists():
            n = copy_images(nonwound_dir, ds1 / "non_wound")
            print(f"   non_wound : {n} images")
            stats["ds1_nonwound"] = n
        else:
            print("   ⚠️  non_wound/ subfolder not found in Wound_Detection/")
            print("      Create it and add skin disease images "
                  "(Epidermolysis, Haemangiosarcoma, Malignant, Meningitis)")
    else:
        print(f"   ⚠️  {wd_folder} not found — skipping Dataset 1")

    ds1_total = stats.get("ds1_wound", 0) + stats.get("ds1_nonwound", 0)
    if ds1_total > 0:
        timeline.done("Dataset 1 — Binary (wound / non_wound)",
                      f"{ds1_total} images")
    else:
        timeline.fail("Dataset 1 — Binary (wound / non_wound)",
                      "no images copied")

    # ── DATASET 2: Multi-class wound types ──────────────────────────────────
    timeline.start("Dataset 2 — Multi-class Wound Types")
    print("\n📁 Building Dataset 2 — Multi-class Wound Types")
    wt_folder = clean / "Wound_Type"

    if wt_folder.exists():
        class_dirs = sorted([d for d in wt_folder.iterdir() if d.is_dir()])
        if class_dirs:
            for cls_dir in class_dirs:
                n = copy_images(cls_dir, ds2 / cls_dir.name)
                print(f"   {cls_dir.name:<25} {n:>5} images")
                stats[f"ds2_{cls_dir.name}"] = n
        else:
            # Flat structure — all images in Wound_Type directly
            imgs = get_images(wt_folder)
            print(f"   ⚠️  No subfolders found in Wound_Type/")
            print(f"      Found {len(imgs)} images directly in folder")
    else:
        print(f"   ⚠️  {wt_folder} not found — skipping Dataset 2")

    ds2_total = sum(v for k, v in stats.items() if k.startswith("ds2_"))
    if ds2_total > 0:
        timeline.done("Dataset 2 — Multi-class Wound Types",
                      f"{ds2_total} images")
    else:
        timeline.fail("Dataset 2 — Multi-class Wound Types",
                      "no images copied")

    # ── DATASET 3: OOD ────────────────────────────────────────────────────────────
    timeline.start("Dataset 3 — OOD")
    print("\n📁 Building Dataset 3 — OOD (evaluation only)")
    ood_folder = clean / "OOD"

    if ood_folder.exists():
        ood_imgs = get_images(ood_folder)
        if ood_imgs:
            (ds3 / "ood").mkdir(parents=True, exist_ok=True)
            for img in ood_imgs:
                dst = ds3 / "ood" / img.name
                if dst.exists():
                    dst = ds3 / "ood" / f"{img.stem}_{img.parent.name}{img.suffix}"
                shutil.copy2(img, dst)
            print(f"   ood       : {len(ood_imgs)} images")
            stats["ds3_ood"] = len(ood_imgs)
    else:
        print(f"   ⚠️  {ood_folder} not found — skipping Dataset 3")

    ds3_total = stats.get("ds3_ood", 0)
    if ds3_total > 0:
        timeline.done("Dataset 3 — OOD", f"{ds3_total} images")
    else:
        timeline.fail("Dataset 3 — OOD", "no images copied")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)

    for ds_path, label in [(ds1,"Dataset 1 — Binary"),
                            (ds2,"Dataset 2 — Multiclass"),
                            (ds3,"Dataset 3 — OOD")]:
        if ds_path.exists():
            classes = {d.name: len(get_images(d))
                       for d in ds_path.iterdir() if d.is_dir()}
            total = sum(classes.values())
            print(f"\n  {label}  (total: {total})")
            for cls, n in sorted(classes.items(), key=lambda x: -x[1]):
                bar = "█" * (n // 5)
                print(f"    {cls:<28} {n:>5}  {bar}")

    # ── Warnings ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  WARNINGS / ACTION NEEDED")
    print("=" * 60)
    issues = False

    # Check binary has both classes
    w_count  = len(get_images(ds1 / "wound"))    if (ds1/"wound").exists()    else 0
    nw_count = len(get_images(ds1 / "non_wound")) if (ds1/"non_wound").exists() else 0
    if nw_count == 0:
        print("  ⚠️  Dataset 1 has NO non_wound images!")
        print("     Action: Create Wound_Detection/non_wound/ and add")
        print("     Epidermolysis, Haemangiosarcoma, Malignant, Meningitis images")
        issues = True
    elif nw_count < 20:
        print(f"  ⚠️  non_wound only has {nw_count} images — very few for training")
        issues = True
    else:
        print(f"  ✅ Dataset 1: wound={w_count}  non_wound={nw_count}")

    # Check multiclass for tiny classes
    if ds2.exists():
        for cls_dir in ds2.iterdir():
            if cls_dir.is_dir():
                n = len(get_images(cls_dir))
                if n < 30:
                    print(f"  ⚠️  {cls_dir.name} only has {n} images "
                          f"— may underperform")
                    issues = True

    if not issues:
        print("  ✅ All datasets look good!")

    # Save report
    timeline.start("Save report")
    with open(out / "dataset_info.json", "w") as f:
        json.dump({"stats": stats}, f, indent=2)
    timeline.done("Save report", str(out / "dataset_info.json"))

    print(f"\n  ✅ Datasets ready in: {out}")
    print(f"\n  ▶  Next step:")
    print(f"     python Train_cpu.py --data_dir ./datasets "
          f"--output_dir ./results\n")

    timeline.save(out / "timeline.json")
    timeline.print_summary()


if __name__ == "__main__":
    main()
