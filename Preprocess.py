"""
WoundAI — Medical Image Preprocessing Pipeline (FIXED)
=======================================================
Handles your exact folder structure:

    MAD/
    ├── OOD/
    │   ├── subfolder_or_images/
    ├── Wound_Detection/
    │   ├── wound/
    │   └── non_wound/
    └── Wound_Type/
        ├── burn/
        ├── laceration/
        └── ...

Run from inside MAD/ folder:
    python Preprocess.py
    python Preprocess.py --raw_dir . --out_dir ./datasets_clean

Steps applied to every image:
  1.  Corrupt / unreadable file check
  2.  Minimum size validation
  3.  Force 3-channel RGB
  4.  Perceptual hash deduplication
  5.  Border artifact crop (4px each edge)
  6.  CLAHE contrast enhancement (LAB colourspace)
  7.  Reinhard colour normalisation
  8.  Letterbox resize → 224×224 (aspect-ratio preserving)
  9.  Blur + exposure quality flagging
  10. Save + full report
"""

import os
import cv2
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from datetime import datetime

try:
    import imagehash
    from PIL import Image as PILImage
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False
    warnings.warn("imagehash not installed — dedup skipped. "
                  "Run: pip install imagehash Pillow")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ─── CONFIG ───────────────────────────────────────────────────────────────────
class Cfg:
    TARGET_SIZE      = 224
    MIN_SIZE_PX      = 64
    MIN_FILE_BYTES   = 1_500
    BLUR_THRESHOLD   = 40.0
    BLUR_REJECT      = False     # True = delete blurry, False = keep + flag
    DARK_THRESHOLD   = 25
    BRIGHT_THRESHOLD = 230
    EXP_REJECT       = False
    PHASH_BITS       = 16
    PHASH_DIST       = 8         # hamming distance threshold for "same image"
    CLAHE_CLIP       = 2.5
    CLAHE_GRID       = (8, 8)
    BORDER_PX        = 4
    JPEG_QUALITY     = 95


cfg = Cfg()


# ─── TIMELINE ─────────────────────────────────────────────────────────────────
class Timeline:
    """Lightweight task-progress tracker with console + JSON output."""

    def __init__(self, name: str):
        self.name   = name
        self.tasks  = []
        self._t0    = {}
        self._wall0 = datetime.now()

    def start(self, task: str):
        self._t0[task] = datetime.now()
        ts = self._t0[task].strftime("%H:%M:%S")
        print(f"  \u23f1  [{ts}] START   {task}")

    def done(self, task: str, note: str = ""):
        elapsed = (datetime.now() - self._t0.get(task, datetime.now())).total_seconds()
        ts = datetime.now().strftime("%H:%M:%S")
        msg = f"  \u2705 [{ts}] DONE    {task}  ({elapsed:.1f}s)"
        if note:
            msg += f"  \u2014 {note}"
        print(msg)
        self.tasks.append({"task": task, "status": "done", "elapsed_s": round(elapsed, 2), "note": note})

    def fail(self, task: str, note: str = ""):
        elapsed = (datetime.now() - self._t0.get(task, datetime.now())).total_seconds()
        ts = datetime.now().strftime("%H:%M:%S")
        msg = f"  \u274c [{ts}] FAILED  {task}  ({elapsed:.1f}s)"
        if note:
            msg += f"  \u2014 {note}"
        print(msg)
        self.tasks.append({"task": task, "status": "failed", "elapsed_s": round(elapsed, 2), "note": note})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        total = (datetime.now() - self._wall0).total_seconds()
        data  = {"pipeline": self.name, "total_elapsed_s": round(total, 2), "tasks": self.tasks}
        path.write_text(json.dumps(data, indent=2))
        print(f"  \U0001f4cb Timeline saved \u2192 {path}")

    def print_summary(self):
        total = (datetime.now() - self._wall0).total_seconds()
        bar   = "\u2500" * 60
        print(f"\n{bar}")
        print(f"  TIMELINE SUMMARY  \u2014  {self.name}")
        print(bar)
        for t in self.tasks:
            icon = "\u2705" if t["status"] == "done" else "\u274c"
            line = f"  {icon} {t['task']}  ({t['elapsed_s']}s)"
            if t.get("note"):
                line += f"  \u2014 {t['note']}"
            print(line)
        print(bar)
        print(f"  Total elapsed: {total:.1f}s  ({total/60:.1f} min)")
        print(f"{bar}\n")



# ─── AUTO-DETECT PROJECT ROOT ─────────────────────────────────────────────────
def find_root(start: Path) -> Path:
    for candidate in [start, start.parent, start.parent.parent]:
        if any((candidate / d).exists()
               for d in ["Wound_Detection", "Wound_Type", "OOD"]):
            return candidate
    return start


# ─── IMAGE PROCESSING STEPS ───────────────────────────────────────────────────
def apply_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE on L channel — enhances wound tissue visibility."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cfg.CLAHE_CLIP, tileGridSize=cfg.CLAHE_GRID)
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def reinhard_normalise(img: np.ndarray) -> np.ndarray:
    """Reinhard colour normalisation in LAB space."""
    TARGET_MEAN = np.array([65.0, 10.0,  8.0])
    TARGET_STD  = np.array([25.0,  8.0, 10.0])
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean = lab.reshape(-1, 3).mean(0)
    src_std  = lab.reshape(-1, 3).std(0) + 1e-6
    lab_norm = (lab - src_mean) / src_std * TARGET_STD + TARGET_MEAN
    return cv2.cvtColor(
        np.clip(lab_norm, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )


def letterbox(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Aspect-ratio-preserving resize with black padding."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    yo, xo = (size - nh) // 2, (size - nw) // 2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    return canvas


def quality_flags(img: np.ndarray) -> list[str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = []
    blur  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean  = float(gray.mean())
    std   = float(gray.std())
    if blur < cfg.BLUR_THRESHOLD:
        flags.append(f"blurry({blur:.0f})")
    if mean < cfg.DARK_THRESHOLD:
        flags.append(f"dark({mean:.0f})")
    elif mean > cfg.BRIGHT_THRESHOLD:
        flags.append(f"overexp({mean:.0f})")
    if std < 5:
        flags.append("blank")
    return flags


class DedupTracker:
    def __init__(self):
        self._seen: dict[str, str] = {}

    def is_dup(self, img_bgr: np.ndarray, path: str) -> tuple[bool, str]:
        if not PHASH_AVAILABLE:
            return False, ""
        pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        h   = str(imagehash.phash(pil, hash_size=cfg.PHASH_BITS))
        for seen_h, seen_p in self._seen.items():
            dist = bin(int(h, 16) ^ int(seen_h, 16)).count("1")
            if dist <= cfg.PHASH_DIST:
                return True, seen_p
        self._seen[h] = path
        return False, ""


def process_one(src: Path, dst: Path, dedup: DedupTracker) -> dict:
    """Run the full pipeline on a single image. Returns status dict."""
    r = {"src": str(src), "dst": str(dst), "ok": True, "reason": "", "flags": []}

    # File size
    try:
        if src.stat().st_size < cfg.MIN_FILE_BYTES:
            return {**r, "ok": False, "reason": "file_too_small"}
    except Exception:
        return {**r, "ok": False, "reason": "stat_error"}

    # Load
    img = cv2.imread(str(src))
    if img is None:
        # Try PIL fallback for unusual formats
        try:
            pil = PILImage.open(src).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return {**r, "ok": False, "reason": "unreadable"}

    # Force 3-channel
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    if min(h, w) < cfg.MIN_SIZE_PX:
        return {**r, "ok": False, "reason": f"too_small({w}x{h})"}

    # Quality flags (before enhancement — check original)
    flags = quality_flags(img)
    r["flags"] = flags

    # Reject conditions
    if "blank" in str(flags):
        return {**r, "ok": False, "reason": "blank_image"}
    if cfg.BLUR_REJECT and any("blurry" in f for f in flags):
        return {**r, "ok": False, "reason": "blurry"}
    if cfg.EXP_REJECT and any(f in str(flags) for f in ["dark", "overexp"]):
        return {**r, "ok": False, "reason": "bad_exposure"}

    # Dedup
    is_d, dup_of = dedup.is_dup(img, str(src))
    if is_d:
        return {**r, "ok": False, "reason": f"duplicate_of:{Path(dup_of).name}"}

    # ── Enhancement pipeline ──────────────────────────────────────────────────
    img = img[cfg.BORDER_PX:-cfg.BORDER_PX,
              cfg.BORDER_PX:-cfg.BORDER_PX]       # border crop
    img = apply_clahe(img)                          # contrast enhancement
    img = reinhard_normalise(img)                   # colour normalisation
    img = letterbox(img, cfg.TARGET_SIZE)           # resize 224×224

    # Save
    save_p = dst.parent / (dst.stem + ".jpg")
    save_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_p), img, [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY])
    r["dst"] = str(save_p)
    return r


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="WoundAI Preprocessing — run from inside MAD/ folder"
    )
    parser.add_argument("--raw_dir",     default=None,
                        help="Project root (auto-detected if not given)")
    parser.add_argument("--out_dir",     default="./datasets_clean")
    parser.add_argument("--blur_reject", action="store_true")
    parser.add_argument("--exp_reject",  action="store_true")
    parser.add_argument("--size",        type=int, default=224)
    parser.add_argument("--folders",     default="all",
                        help="Which folders to process: all | Wound_Type | "
                             "Wound_Detection | OOD  (comma-separated)")
    args = parser.parse_args()

    cfg.BLUR_REJECT  = args.blur_reject
    cfg.EXP_REJECT   = args.exp_reject
    cfg.TARGET_SIZE  = args.size

    raw_root = Path(args.raw_dir) if args.raw_dir else find_root(Path.cwd())
    out_root = Path(args.out_dir)

    # Which top-level folders to process
    target_folders = (
        ["Wound_Detection", "Wound_Type", "OOD"]
        if args.folders == "all"
        else [f.strip() for f in args.folders.split(",")]
    )

    print("=" * 65)
    print("  WoundAI — Medical Image Preprocessing")
    print("=" * 65)
    print(f"  Source : {raw_root}")
    print(f"  Output : {out_root}")
    print(f"  Size   : {cfg.TARGET_SIZE}×{cfg.TARGET_SIZE}")
    print(f"  CLAHE  : ON  (clip={cfg.CLAHE_CLIP})")
    print(f"  ColorN : ON  (Reinhard LAB)")
    print(f"  Dedup  : {'ON (pHash)' if PHASH_AVAILABLE else 'OFF'}")
    print(f"  Reject blurry: {cfg.BLUR_REJECT}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65 + "\n")

    timeline = Timeline("WoundAI Preprocessing")

    # Validate source
    active_folders = []
    timeline.start("Scan source folders")
    for fname in target_folders:
        fpath = raw_root / fname
        if fpath.exists():
            active_folders.append(fpath)
            imgs = list(fpath.rglob("*"))
            imgs = [f for f in imgs if f.is_file() and f.suffix.lower() in IMG_EXTS]
            print(f"  📁 {fname:<25} {len(imgs):>5} images")
        else:
            print(f"  ⚠️  NOT FOUND: {fpath}")

    if not active_folders:
        timeline.fail("Scan source folders", f"no folders found under {raw_root}")
        print(f"\n❌ No folders found under {raw_root}")
        print("   Make sure you're running from inside your MAD/ folder,")
        print("   or pass --raw_dir /full/path/to/MAD")
        return
    timeline.done("Scan source folders", f"{len(active_folders)} folder(s) ready")

    # ── Process ───────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    dedup = DedupTracker()

    for top_folder in active_folders:
        print(f"\n  Processing {top_folder.name}/...")
        timeline.start(f"Process {top_folder.name}")

        # Find class subfolders (or treat top_folder as single class)
        class_dirs = sorted([d for d in top_folder.iterdir() if d.is_dir()])
        if not class_dirs:
            # Flat: images directly in top_folder
            class_dirs = [top_folder]

        for cls_dir in class_dirs:
            imgs = [f for f in cls_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMG_EXTS]
            if not imgs:
                continue

            ok_count = 0
            for img_path in tqdm(imgs, desc=f"    {cls_dir.name[:28]}", unit="img"):
                # Mirror folder structure under out_root
                try:
                    rel = img_path.relative_to(raw_root)
                except ValueError:
                    rel = Path(top_folder.name) / cls_dir.name / img_path.name

                dst = out_root / rel.parent / rel.stem
                result = process_one(img_path, dst, dedup)
                all_results.append(result)
                if result["ok"]:
                    ok_count += 1

            rejected = len(imgs) - ok_count
            tqdm.write(f"    ✅ {cls_dir.name:<28} "
                       f"{ok_count:>4} saved  {rejected:>3} rejected")

        folder_saved = sum(1 for r in all_results if r['ok'])
        timeline.done(f"Process {top_folder.name}",
                      f"{folder_saved} images saved")

    # ── Summary ───────────────────────────────────────────────────────────────
    total     = len(all_results)
    saved     = sum(1 for r in all_results if r["ok"])
    rejected  = total - saved
    dups      = sum(1 for r in all_results if "duplicate" in r.get("reason",""))
    blurry    = sum(1 for r in all_results if "blurry"    in r.get("reason",""))
    dark      = sum(1 for r in all_results if "dark"      in r.get("reason",""))
    blank     = sum(1 for r in all_results if "blank"     in r.get("reason",""))
    unread    = sum(1 for r in all_results if "unreadable" in r.get("reason",""))
    flagged   = sum(1 for r in all_results if r["ok"] and r["flags"])

    pct_s = (saved    / total * 100) if total else 0
    pct_r = (rejected / total * 100) if total else 0

    print("\n" + "=" * 65)
    print("  PREPROCESSING COMPLETE")
    print("=" * 65)
    print(f"  Total processed      : {total:>6}")
    print(f"  ✅ Saved (clean)     : {saved:>6}  ({pct_s:.1f}%)")
    print(f"  ⚠️  Flagged (kept)    : {flagged:>6}  (blur/exposure warnings)")
    print(f"  ❌ Rejected total    : {rejected:>6}  ({pct_r:.1f}%)")
    print(f"     ├─ Duplicates     : {dups:>6}")
    print(f"     ├─ Blurry         : {blurry:>6}")
    print(f"     ├─ Too dark       : {dark:>6}")
    print(f"     ├─ Blank/corrupt  : {blank:>6}")
    print(f"     └─ Unreadable     : {unread:>6}")

    # Per-output-folder counts
    print("\n  Output folder counts:")
    out_counts: Counter = Counter()
    for r in all_results:
        if r["ok"]:
            # e.g. datasets_clean/Wound_Type/burn/img.jpg → Wound_Type/burn
            parts = Path(r["dst"]).parts
            try:
                idx = next(i for i, p in enumerate(parts)
                           if p in ["Wound_Type","Wound_Detection","OOD"])
                key = "/".join(parts[idx:idx+2])
            except StopIteration:
                key = Path(r["dst"]).parent.name
            out_counts[key] += 1

    for folder, count in sorted(out_counts.items()):
        bar = "█" * (count // 3)
        print(f"    {folder:<40} {count:>5}  {bar}")

    # Save report
    timeline.start("Save report")
    report_path = out_root / "_review" / "preprocessing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "total": total, "saved": saved, "rejected": rejected,
            "duplicates": dups, "blurry": blurry, "dark": dark,
            "blank": blank, "unreadable": unread, "flagged": flagged,
            "output_counts": dict(out_counts),
            "rejected_list": [r for r in all_results if not r["ok"]]
        }, f, indent=2)
    timeline.done("Save report", str(report_path))

    timeline.save(out_root / "_review" / "timeline.json")
    timeline.print_summary()


if __name__ == "__main__":
    main()
