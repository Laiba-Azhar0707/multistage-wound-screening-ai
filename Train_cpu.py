"""
WoundAI — Multi-Model Comparison Pipeline (Edge-Optimized)
===========================================================
Models compared:
    1. Custom Lightweight CNN     (from scratch)
    2. MobileNetV2                (best for edge)
    3. ResNet50V2                 (accuracy benchmark)
    4. EfficientNetB0             (best accuracy/size tradeoff)
    5. EfficientNetB3             (heavier, more accurate)

All 3 Stages:
    Stage 1 → Binary wound detector        (wound / non_wound)
    Stage 2 → Multi-class wound classifier  (12 wound types, 5-Fold CV)
    Stage 3 → OOD robustness evaluation

Final → Comparison table with accuracy, F1, AUC, inference speed, model size

Run:
    pip install tensorflow scikit-learn matplotlib seaborn
    python train_multimodel.py --data_dir ./datasets --output_dir ./results
"""

import os
import sys
import io
import json
import time
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Force UTF-8 stdout/stderr so emoji in print() never crash on Windows cp1252
# consoles, Tee-Object pipes, or redirected log files.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import Counter
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50V2, EfficientNetB0, EfficientNetB3
)
from tensorflow.keras.applications import (
    mobilenet_v2, resnet_v2, efficientnet
)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE   = 96       # balanced for edge devices
BATCH_SIZE = 16
EPOCHS     = 50
SEED       = 42
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Mapping from datasets_clean/Wound_Detection/Wound sub-folder names
# to canonical Stage-2 class names.  None = skip (non-wound or too few images).
WOUND_DETECTION_MAP: dict = {
    # ── Identical to existing classes ──────────────────────────────────────
    "Abrasion":               "Abrasion",
    "abrasions":              "Abrasion",
    "Bruise":                 "Bruise",
    "Bruises":                "Bruise",
    "Burn":                   "Burn",
    "Burns":                  "Burn",
    "Burns_and_scalds":       "Burn",
    "cut":                    "cut",
    "Cuts":                   "cut",
    "Diabetic_ulcer":         "Diabetic_ulcer",
    "Diabetic_footulcers":    "Diabetic_ulcer",
    "Infected_wound":         "Infected_wound",
    "Infected_toes":          "Infected_wound",
    "Extravasation_injuries": "Infected_wound",
    "Laceration":             "Laceration",
    "lacerations":            "Laceration",
    "Stab_wound":             "Laceration",
    "Pressure_ulcer":         "Pressure_ulcer",
    "Pressure_ulcers":        "Pressure_ulcer",
    "Venous_ulcer":           "Venous_ulcer",
    "Venous_Arterial_ulcers": "Venous_ulcer",
    # ── NEW distinct wound types ────────────────────────────────────────────
    "Pressure_wound":         "Pressure_wound",
    "Surgical_wound":         "Surgical_wound",
    "Venous_wound":           "Venous_wound",
    # ── Skip (not a wound type / too few samples) ───────────────────────────
    "Pilonidal_sinus_wounds": None,
    "Ingrown_nails":          None,
}

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── GPU SETUP ────────────────────────────────────────────────────────────────
def _is_directml() -> bool:
    """
    Return True when the active GPU backend is DirectML (Windows DX12).
    Strategy: the most reliable signal is whether tensorflow-directml-plugin
    is installed in the current Python environment.
    """
    # 1. Check installed packages (most reliable — plugin always surfaces here)
    try:
        import importlib.metadata as _meta
        pkgs = {d.metadata["Name"].lower() for d in _meta.distributions()}
        if "tensorflow-directml-plugin" in pkgs or "tensorflow-directml" in pkgs:
            return True
    except Exception:
        pass

    # 2. Fallback: scan site-packages for the directml dll
    try:
        import site, glob
        for sp in site.getsitepackages():
            if glob.glob(sp + "/**/directml*.dll", recursive=True):
                return True
    except Exception:
        pass

    # 3. Fallback: TF device list string
    try:
        from tensorflow.python.eager import context as _ctx
        for d in _ctx.context().list_devices():
            if "directml" in str(d).lower():
                return True
    except Exception:
        pass

    return False


def setup_gpu(force_cpu: bool = False) -> bool:
    """
    Configure GPU for training.

    • Enables memory growth on all physical GPUs.
    • Auto-detects the DirectML backend (Windows DX12) and skips
      mixed_float16 — DirectML does NOT support float16 compute ops and
      will crash with: 'ImageProjectiveTransformV3 is not supported by DML'.
    • On native CUDA GPUs, enables mixed_float16 for ~2× throughput.
    • Returns True if a GPU device is available and will be used.
    """
    if force_cpu:
        print("  💻  --no_gpu set — forcing CPU (float32)")
        # Hide all GPUs so TF falls back to CPU
        tf.config.set_visible_devices([], "GPU")
        return False

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("  💻  No GPU found — running on CPU (float32)")
        return False

    # Enable memory growth (must be done before any GPU op)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # already initialised

    print(f"  🖥  GPU(s) detected : {len(gpus)}")
    for g in gpus:
        print(f"       {g.name}")

    # Detect DirectML (Windows DX12 backend)
    directml = _is_directml()
    if directml:
        # DirectML does NOT support float16 compute — stay on float32
        tf.keras.mixed_precision.set_global_policy("float32")
        print("  🎮  Backend         : DirectML (Windows DirectX 12)")
        print("  ℹ️   Mixed precision : DISABLED (DirectML float16 not supported)")
    else:
        # Native CUDA — enable mixed_float16 for Ampere/Turing speed boost
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("  ⚡  Backend         : CUDA")
        print("  ⚡  Mixed precision : float16 (compute) + float32 (weights)")

    return True


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


# ─── MODEL REGISTRY ───────────────────────────────────────────────────────────
# Each entry: (display_name, builder_fn, preprocess_fn)
MODEL_REGISTRY = {}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_custom_cnn(num_classes: int, img_size: int = IMG_SIZE):
    """
    Lightweight custom CNN designed for edge deployment.
    Uses depthwise separable convolutions to stay fast and small.
    """
    inp = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inp)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2 — depthwise separable
    x = layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 1, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3 — depthwise separable
    x = layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 1, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 4
    x = layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 1, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.3)(x)

    if num_classes == 1:
        out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    else:
        out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return models.Model(inp, out, name="CustomCNN")


def build_transfer_model(base_class, preprocess_fn,
                         model_name: str, num_classes: int,
                         img_size: int = IMG_SIZE,
                         fine_tune_layers: int = 20):
    """
    Generic transfer learning builder for MobileNetV2, ResNet50V2,
    EfficientNetB0, EfficientNetB3.
    Fine-tunes the last `fine_tune_layers` layers for better accuracy.
    """
    inp  = layers.Input(shape=(img_size, img_size, 3))
    x    = preprocess_fn(inp)

    base = base_class(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    # Freeze all first, then unfreeze top N for fine-tuning
    base.trainable = False
    if fine_tune_layers > 0:
        for layer in base.layers[-fine_tune_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.3)(x)

    if num_classes == 1:
        out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    else:
        out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return models.Model(inp, out, name=model_name)


def get_all_models(num_classes: int):
    """Returns dict of {model_name: model_instance}"""
    return {
        "CustomCNN": build_custom_cnn(num_classes),

        "MobileNetV2": build_transfer_model(
            MobileNetV2,
            mobilenet_v2.preprocess_input,
            "MobileNetV2", num_classes,
            fine_tune_layers=20
        ),

        "ResNet50V2": build_transfer_model(
            ResNet50V2,
            resnet_v2.preprocess_input,
            "ResNet50V2", num_classes,
            fine_tune_layers=30
        ),

        "EfficientNetB0": build_transfer_model(
            EfficientNetB0,
            efficientnet.preprocess_input,
            "EfficientNetB0", num_classes,
            fine_tune_layers=20
        ),

        "EfficientNetB3": build_transfer_model(
            EfficientNetB3,
            efficientnet.preprocess_input,
            "EfficientNetB3", num_classes,
            fine_tune_layers=30
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_augmenter():
    # Only RandomFlip is safe on DirectML — Zoom/Translation/Contrast/Brightness
    # all trigger unsupported ops (ImageProjectiveTransformV3, AdjustContrastv2)
    # that cause fatal crashes on the DirectML GPU backend.
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
    ], name="augmentation")


def load_folder(data_dir: Path):
    """Load image paths + integer labels from class subfolders."""
    cls_dirs    = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in cls_dirs]
    paths, labels = [], []

    print(f"\n  Classes in {data_dir.name}:")
    for idx, cls_dir in enumerate(cls_dirs):
        imgs = [str(f) for f in cls_dir.rglob("*")
                if f.suffix.lower() in IMG_EXTS]
        print(f"    [{idx}] {cls_dir.name:<25} {len(imgs):>5} images")
        paths.extend(imgs)
        labels.extend([idx] * len(imgs))

    print(f"    {'TOTAL':<30} {len(paths):>5}")
    return paths, labels, class_names


def load_wound_detection_extra(
        base_dir: Path,
        existing_class_names: list,
        class_map: dict = WOUND_DETECTION_MAP,
) -> tuple:
    """
    Scans datasets_clean/Wound_Detection/Wound/ and returns extra
    (paths, labels, updated_class_names) using WOUND_DETECTION_MAP.

    * Folders whose canonical name already exists in existing_class_names
      get the matching label index (extends existing class data).
    * Folders whose canonical name is NEW get appended as a new class.
    * Folders mapped to None are silently skipped.
    """
    wound_dir = base_dir / "datasets_clean" / "Wound_Detection" / "Wound"
    if not wound_dir.exists():
        print(f"  ⚠  Extra wound folder not found: {wound_dir}")
        return [], [], existing_class_names

    # Work with a mutable copy so we can append new class names
    names = list(existing_class_names)
    extra_paths, extra_labels = [], []
    skipped, added = [], []

    for sub in sorted(wound_dir.iterdir()):
        if not sub.is_dir():
            continue
        canonical = class_map.get(sub.name)
        if canonical is None:
            skipped.append(sub.name)
            continue

        # Resolve or create label index
        if canonical not in names:
            names.append(canonical)
        idx = names.index(canonical)

        imgs = [str(f) for f in sub.rglob("*")
                if f.suffix.lower() in IMG_EXTS]
        extra_paths.extend(imgs)
        extra_labels.extend([idx] * len(imgs))
        added.append(f"{sub.name} → {canonical} ({len(imgs)} imgs)")

    print(f"\n  [Extra data from Wound_Detection/Wound/]")
    for a in added:
        print(f"    + {a}")
    if skipped:
        print(f"    ~ Skipped: {', '.join(skipped)}")

    return extra_paths, extra_labels, names


def load_image(path: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32)


def build_tf_dataset(paths, labels, augment=False, batch_size=None,
                     drop_remainder=False):
    """Builds a tf.data.Dataset with optional augmentation.

    batch_size defaults to the global BATCH_SIZE resolved at call-time
    (avoids Python's early-binding of default arguments capturing the
    pre-GPU-bump value of 16 instead of the runtime value of 32).
    drop_remainder=True is recommended for training sets so that every
    batch is exactly the same size — important for BatchNorm layers and
    avoids the tiny last-batch GPU dispatch that can trigger
    DXGI_ERROR_DEVICE_HUNG on DirectML.
    prefetch(2) instead of AUTOTUNE prevents the driver from queuing too
    many compute ops at once, which also contributes to device hangs.
    """
    if batch_size is None:
        batch_size = BATCH_SIZE  # resolved at call-time, picks up GPU bump

    augmenter = build_augmenter() if augment else None

    def process(path, label):
        img = load_image(path)
        if augment:
            img = augmenter(tf.expand_dims(img, 0), training=True)[0]
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(2)
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class SanitizeLogsCallback(tf.keras.callbacks.Callback):
    """
    Converts EagerTensors and numpy scalars/arrays in the Keras logs dict to
    plain Python floats/ints **in-place** before ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, and CSVLogger see them.
    Also replaces NaN/Inf values with 0.0 to prevent GPU hangs (DXGI_ERROR_DEVICE_HUNG)
    caused by overflowed metrics propagating back to the DirectML device.
    """
    _SAFE_MAX = 1e6   # anything larger is treated as overflow

    @staticmethod
    def _to_python(v):
        if hasattr(v, 'numpy'):
            v = v.numpy()
        if isinstance(v, np.ndarray):
            return v.item() if v.ndim == 0 else v.tolist()
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v

    def _sanitize(self, logs):
        if logs:
            for k in list(logs.keys()):
                v = self._to_python(logs[k])
                # Replace NaN, Inf, or absurd overflow values with 0.0
                if isinstance(v, float) and (v != v or abs(v) == float('inf')
                                             or abs(v) > self._SAFE_MAX):
                    v = 0.0
                logs[k] = v

    def on_epoch_end(self, epoch, logs=None):        self._sanitize(logs)
    def on_train_batch_end(self, batch, logs=None): self._sanitize(logs)
    def on_test_batch_end(self, batch, logs=None):  self._sanitize(logs)


def get_callbacks(save_path: str, monitor: str, mode: str, log_path: str):
    return [
        SanitizeLogsCallback(),                        # ← must be first
        callbacks.TerminateOnNaN(),
        callbacks.ModelCheckpoint(save_path, monitor=monitor,
                                  save_best_only=True, mode=mode, verbose=0),
        callbacks.EarlyStopping(monitor=monitor, patience=12,
                                restore_best_weights=True, mode=mode),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-7, verbose=0),
        callbacks.CSVLogger(log_path)
    ]


def measure_inference_speed(model, num_samples=100):
    """Measures average inference time per image in milliseconds."""
    dummy = np.random.rand(num_samples, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    # Warmup
    model.predict(dummy[:4], verbose=0)
    t0 = time.time()
    model.predict(dummy, batch_size=1, verbose=0)
    elapsed = (time.time() - t0) / num_samples * 1000
    return round(elapsed, 2)


def get_model_size_mb(model):
    """Returns trainable parameter count and estimated size in MB."""
    params = model.count_params()
    size_mb = round(params * 4 / 1e6, 2)  # float32 = 4 bytes
    return params, size_mb


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_history(h, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)
    for ax, metric, val_metric in zip(
            axes, ["accuracy", "loss"], ["val_accuracy", "val_loss"]):
        ax.plot(h.history.get(metric, []), label="Train")
        ax.plot(h.history.get(val_metric, []), label="Val")
        ax.set(title=metric.capitalize(), xlabel="Epoch")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusion(y_true, y_pred, names, path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, len(names)*1.2), max(5, len(names))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names)
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_model_comparison(results: dict, out_dir: Path, stage: str):
    """Bar chart comparing all models on key metrics."""
    model_names = list(results.keys())
    metrics = ["accuracy", "f1", "inference_ms", "size_mb"]
    titles  = ["Accuracy", "Macro F1", "Inference (ms/img)", "Model Size (MB)"]
    colors  = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{stage} — Model Comparison", fontsize=14, fontweight="bold")

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        vals = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(model_names, vals, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(title)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f"{v:.3f}" if metric not in ["inference_ms", "size_mb"] else f"{v}",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = out_dir / f"{stage.lower().replace(' ', '_')}_comparison.png"
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"  📊 Comparison chart saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — BINARY WOUND DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def run_stage1(datasets_dir: Path, out_dir: Path):
    print("\n" + "=" * 65)
    print("  STAGE 1 — Binary Wound Detector  (wound vs non_wound)")
    print("=" * 65)

    data_dir = datasets_dir / "dataset1_binary"
    if not data_dir.exists():
        print(f"  ❌ Not found: {data_dir}")
        return {}

    paths, labels, class_names = load_folder(data_dir)
    if len(set(labels)) < 2:
        print("  ❌ Need both 'wound' and 'non_wound' folders")
        return {}

    # Split
    Xtr, Xte, ytr, yte = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr, ytr, test_size=0.18, stratify=ytr, random_state=SEED)

    print(f"\n  Split → Train:{len(Xtr)}  Val:{len(Xva)}  Test:{len(Xte)}")

    # Class weights
    cw = compute_class_weight("balanced", classes=np.array([0,1]), y=np.array(labels))
    cw_dict = {0: float(cw[0]), 1: float(cw[1])}

    # Build datasets
    ds_train = build_tf_dataset(Xtr, ytr, augment=True,  drop_remainder=True)
    ds_val   = build_tf_dataset(Xva, yva, augment=False, drop_remainder=False)
    ds_test  = build_tf_dataset(Xte, yte, augment=False, drop_remainder=False)

    s1_dir = out_dir / "stage1"
    s1_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name, model in get_all_models(num_classes=1).items():
        print(f"\n  ── Training: {model_name} ──")
        params, size_mb = get_model_size_mb(model)
        print(f"     Params: {params:,}  |  Size: {size_mb} MB")

        model_dir = s1_dir / model_name
        model_dir.mkdir(exist_ok=True)

        model.compile(
            optimizer=optimizers.Adam(1e-3, clipnorm=1.0),
            loss="binary_crossentropy",
            metrics=["accuracy",
                     tf.keras.metrics.AUC(name="auc"),
                     tf.keras.metrics.Recall(name="recall"),
                     tf.keras.metrics.Precision(name="precision")]
        )

        cb = get_callbacks(
            str(model_dir / "best.keras"),
            monitor="val_auc", mode="max",
            log_path=str(model_dir / "log.csv")
        )

        t0 = time.time()
        h = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=EPOCHS,
            class_weight=cw_dict,
            callbacks=cb,
            verbose=1
        )
        train_time = round((time.time() - t0) / 60, 1)
        print(f"     ✅ Trained in {train_time} min")

        plot_history(h, str(model_dir / "curves.png"),
                     f"Stage 1 — {model_name}")

        # Evaluate
        y_true, y_prob = [], []
        for imgs, lbls in ds_test:
            probs = model.predict(imgs, verbose=0).flatten()
            y_prob.extend(probs.tolist())
            y_true.extend(lbls.numpy().tolist())

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob >= 0.5).astype(int)

        acc  = float(accuracy_score(y_true, y_pred))
        f1   = float(f1_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.0

        inf_ms = measure_inference_speed(model)

        print(f"     Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
              f"Inf={inf_ms}ms  Size={size_mb}MB")

        plot_confusion(y_true, y_pred, class_names,
                       str(model_dir / "confusion.png"),
                       f"Stage 1 — {model_name}")

        all_results[model_name] = {
            "accuracy": acc, "f1": f1, "precision": prec,
            "recall": rec, "auc_roc": auc,
            "inference_ms": inf_ms, "size_mb": size_mb,
            "params": params, "train_time_min": train_time
        }

    # Save + plot comparison
    with open(s1_dir / "all_results.json", "w") as f:
        json.dump({"class_names": class_names, "models": all_results}, f, indent=2)

    plot_model_comparison(all_results, s1_dir, "Stage 1 Binary Detection")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — MULTI-CLASS WOUND CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def run_stage2(datasets_dir: Path, out_dir: Path):
    print("\n" + "=" * 65)
    print("  STAGE 2 — Multi-class Wound Classifier  (5-Fold CV)")
    print("=" * 65)

    data_dir = datasets_dir / "dataset2_multiclass"
    if not data_dir.exists():
        print(f"  ❌ Not found: {data_dir}")
        return {}

    paths, labels, class_names = load_folder(data_dir)

    # ── Merge extra wounds from datasets_clean/Wound_Detection/Wound/ ──────
    # Pass the project root (parent of 'datasets/') so the helper can locate
    # datasets_clean/ regardless of how --data_dir is set.
    project_root = datasets_dir.parent
    ep, el, class_names = load_wound_detection_extra(
        project_root, class_names, WOUND_DETECTION_MAP)
    paths  = paths  + ep
    labels = labels + el
    print(f"\n  Combined dataset: {len(paths)} images, "
          f"{len(class_names)} classes: {class_names}")
    # ────────────────────────────────────────────────────────────────────────

    num_classes = len(class_names)
    if num_classes < 2:
        print("  ❌ Need at least 2 wound type classes")
        return {}

    cw = compute_class_weight("balanced",
                               classes=np.arange(num_classes), y=np.array(labels))
    cw_dict = {i: float(w) for i, w in enumerate(cw)}

    # Train/test split for final eval
    Xtr, Xte, ytr, yte = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=SEED)

    s2_dir = out_dir / "stage2"
    s2_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    paths_arr  = np.array(paths)
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for model_name in get_all_models(num_classes).keys():
        print(f"\n  ── {model_name} ──")
        model_dir = s2_dir / model_name
        model_dir.mkdir(exist_ok=True)

        params, size_mb = get_model_size_mb(
            get_all_models(num_classes)[model_name])

        # 5-Fold CV
        fold_accs, fold_f1s = [], []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(paths_arr, labels_arr)):
            m = get_all_models(num_classes)[model_name]
            m.compile(
                optimizer=optimizers.Adam(1e-3, clipnorm=1.0),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            ds_tr = build_tf_dataset(
                paths_arr[tr_idx].tolist(),
                labels_arr[tr_idx].tolist(), augment=True,  drop_remainder=True)
            ds_va = build_tf_dataset(
                paths_arr[va_idx].tolist(),
                labels_arr[va_idx].tolist(), augment=False, drop_remainder=False)

            cb_fold = [
                callbacks.TerminateOnNaN(),
                callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                                        restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                            patience=4, min_lr=1e-7)
            ]
            m.fit(ds_tr, validation_data=ds_va,
                  epochs=EPOCHS, class_weight=cw_dict,
                  callbacks=cb_fold, verbose=0)

            y_va, y_pr = [], []
            for imgs, lbls in ds_va:
                preds = np.argmax(m.predict(imgs, verbose=0), axis=1)
                y_pr.extend(preds.tolist())
                y_va.extend(lbls.numpy().tolist())

            acc = accuracy_score(y_va, y_pr)
            mf1 = f1_score(y_va, y_pr, average="macro", zero_division=0)
            fold_accs.append(acc)
            fold_f1s.append(mf1)
            print(f"   Fold {fold+1}/5 → Acc={acc:.4f}  Macro-F1={mf1:.4f}")

        cv_acc_mean = float(np.mean(fold_accs))
        cv_acc_std  = float(np.std(fold_accs))
        cv_f1_mean  = float(np.mean(fold_f1s))
        cv_f1_std   = float(np.std(fold_f1s))

        print(f"   CV Accuracy: {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
        print(f"   CV Macro-F1: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

        # Final model on full train set
        final_m = get_all_models(num_classes)[model_name]
        final_m.compile(
            optimizer=optimizers.Adam(8e-4, clipnorm=1.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy",
                     tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")]
        )
        ds_train = build_tf_dataset(Xtr, ytr, augment=True,  drop_remainder=True)
        ds_test  = build_tf_dataset(Xte, yte, augment=False, drop_remainder=False)

        cb_final = get_callbacks(
            str(model_dir / "best.keras"),
            monitor="val_accuracy", mode="max",
            log_path=str(model_dir / "log.csv")
        )
        t0 = time.time()
        h = final_m.fit(ds_train, validation_data=ds_test,
                         epochs=EPOCHS, class_weight=cw_dict,
                         callbacks=cb_final, verbose=1)
        train_time = round((time.time() - t0) / 60, 1)

        plot_history(h, str(model_dir / "curves.png"),
                     f"Stage 2 — {model_name}")

        y_true, y_pred = [], []
        for imgs, lbls in ds_test:
            preds = np.argmax(final_m.predict(imgs, verbose=0), axis=1)
            y_pred.extend(preds.tolist())
            y_true.extend(lbls.numpy().tolist())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = float(accuracy_score(y_true, y_pred))
        f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        inf_ms = measure_inference_speed(final_m)

        print(f"   Final Test → Acc={acc:.4f}  Macro-F1={f1:.4f}  "
              f"Inf={inf_ms}ms  Size={size_mb}MB")

        plot_confusion(y_true, y_pred, class_names,
                       str(model_dir / "confusion.png"),
                       f"Stage 2 — {model_name}")

        all_results[model_name] = {
            "cv_accuracy_mean": cv_acc_mean, "cv_accuracy_std": cv_acc_std,
            "cv_f1_mean": cv_f1_mean, "cv_f1_std": cv_f1_std,
            "accuracy": acc, "f1": f1,
            "inference_ms": inf_ms, "size_mb": size_mb,
            "params": params, "train_time_min": train_time,
            "fold_accs": fold_accs, "fold_f1s": fold_f1s
        }

    with open(s2_dir / "all_results.json", "w") as f:
        json.dump({"class_names": class_names, "models": all_results}, f, indent=2)

    plot_model_comparison(all_results, s2_dir, "Stage 2 Multi-class")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — OOD ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════

def run_stage3(datasets_dir: Path, out_dir: Path,
               s1_results: dict, s2_results: dict):
    print("\n" + "=" * 65)
    print("  STAGE 3 — OOD Robustness Evaluation")
    print("=" * 65)

    ood_dir = datasets_dir / "dataset3_ood"
    if not ood_dir.exists():
        print(f"  ❌ Not found: {ood_dir}")
        return {}

    ood_imgs = [str(f) for f in ood_dir.rglob("*")
                if f.suffix.lower() in IMG_EXTS]
    if not ood_imgs:
        print("  ❌ No OOD images found")
        return {}

    print(f"  OOD images: {len(ood_imgs)}")
    ood_labels = [0] * len(ood_imgs)
    ds_ood = build_tf_dataset(ood_imgs, ood_labels, augment=False, batch_size=16)

    s3_dir = out_dir / "stage3"
    s3_dir.mkdir(parents=True, exist_ok=True)

    ood_results = {}

    # Re-load best Stage 1 models
    s1_model_dir = out_dir / "stage1"
    s2_model_dir = out_dir / "stage2"

    fig, axes = plt.subplots(2, len(s1_results), figsize=(5*len(s1_results), 8))
    if len(s1_results) == 1:
        axes = axes.reshape(2, 1)

    for col, model_name in enumerate(s1_results.keys()):
        print(f"\n  ── {model_name} ──")
        s1_best = s1_model_dir / model_name / "best.keras"
        s2_best = s2_model_dir / model_name / "best.keras"

        if not s1_best.exists() or not s2_best.exists():
            print(f"     ⚠️  Saved models not found, skipping.")
            continue

        m1 = tf.keras.models.load_model(str(s1_best))
        m2 = tf.keras.models.load_model(str(s2_best))

        # Stage 1 OOD
        s1_probs = []
        for imgs, _ in ds_ood:
            s1_probs.extend(m1.predict(imgs, verbose=0).flatten().tolist())
        s1_probs = np.array(s1_probs)
        s1_preds = (s1_probs >= 0.5).astype(int)
        fpr      = float(s1_preds.mean())

        # Stage 2 OOD
        s2_probs = []
        for imgs, _ in ds_ood:
            s2_probs.extend(m2.predict(imgs, verbose=0).tolist())
        s2_probs    = np.array(s2_probs)
        s2_max_conf = s2_probs.max(axis=1)
        high_conf   = float((s2_max_conf > 0.8).mean())

        print(f"     Stage1 FPR={fpr:.4f}  Stage2 HighConf={high_conf:.4f}")

        ood_results[model_name] = {
            "stage1_false_positive_rate": fpr,
            "stage1_mean_confidence": float(s1_probs.mean()),
            "stage2_high_conf_rate": high_conf,
            "stage2_mean_max_conf": float(s2_max_conf.mean())
        }

        # Plot confidence histograms
        axes[0, col].hist(s1_probs, bins=20, color="#ef4444", alpha=0.75, edgecolor="white")
        axes[0, col].axvline(0.5, color="black", linestyle="--")
        axes[0, col].set(title=f"{model_name}\nS1 P(wound) on OOD",
                         xlabel="P(wound)", ylabel="Count")

        axes[1, col].hist(s2_max_conf, bins=20, color="#f59e0b",
                          alpha=0.75, edgecolor="white")
        axes[1, col].axvline(0.8, color="black", linestyle="--")
        axes[1, col].set(title=f"S2 Max Confidence on OOD",
                         xlabel="Confidence", ylabel="Count")

    plt.tight_layout()
    plt.savefig(str(s3_dir / "ood_comparison.png"), dpi=130)
    plt.close()

    with open(s3_dir / "all_results.json", "w") as f:
        json.dump(ood_results, f, indent=2)

    print(f"\n  ✅ Stage 3 saved → {s3_dir}")
    return ood_results


# ══════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_final_table(out_dir: Path):
    print("\n" + "=" * 75)
    print("  FINAL COMPARISON TABLE — All Models × All Stages")
    print("=" * 75)

    s1_path = out_dir / "stage1" / "all_results.json"
    s2_path = out_dir / "stage2" / "all_results.json"
    s3_path = out_dir / "stage3" / "all_results.json"

    if s1_path.exists():
        s1 = json.load(open(s1_path))["models"]
        print(f"\n  ── Table 1: Stage 1 Binary Detection ──")
        header = f"  {'Model':<18} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7} {'ms/img':>8} {'MB':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, r in s1.items():
            print(f"  {name:<18} {r['accuracy']:>7.4f} {r['f1']:>7.4f} "
                  f"{r['auc_roc']:>7.4f} {r['precision']:>7.4f} "
                  f"{r['recall']:>7.4f} {r['inference_ms']:>8} {r['size_mb']:>6}")

    if s2_path.exists():
        s2 = json.load(open(s2_path))["models"]
        print(f"\n  ── Table 2: Stage 2 Multi-class (5-Fold CV) ──")
        header = f"  {'Model':<18} {'CV Acc':>9} {'± Std':>7} {'CV F1':>7} {'± Std':>7} {'ms/img':>8} {'MB':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, r in s2.items():
            print(f"  {name:<18} {r['cv_accuracy_mean']:>9.4f} "
                  f"±{r['cv_accuracy_std']:>6.4f} "
                  f"{r['cv_f1_mean']:>7.4f} ±{r['cv_f1_std']:>6.4f} "
                  f"{r['inference_ms']:>8} {r['size_mb']:>6}")

    if s3_path.exists():
        s3 = json.load(open(s3_path))
        print(f"\n  ── Table 3: Stage 3 OOD Robustness ──")
        header = f"  {'Model':<18} {'S1 FPR':>8} {'S1 Conf':>9} {'S2 HiConf':>10} {'Ideal FPR':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, r in s3.items():
            fpr_ok = "✅" if r["stage1_false_positive_rate"] < 0.10 else "❌"
            print(f"  {name:<18} {r['stage1_false_positive_rate']:>8.4f} "
                  f"{r['stage1_mean_confidence']:>9.4f} "
                  f"{r['stage2_high_conf_rate']:>10.4f}  {fpr_ok} <0.10")

    # Edge deployment recommendation
    print(f"\n  ── Edge Deployment Recommendation ──")
    if s1_path.exists() and s2_path.exists():
        s1 = json.load(open(s1_path))["models"]
        s2 = json.load(open(s2_path))["models"]

        # Score = acc*0.4 + f1*0.4 - normalized_ms*0.2
        scores = {}
        all_ms = [s1[m]["inference_ms"] for m in s1] + [s2[m]["inference_ms"] for m in s2]
        max_ms = max(all_ms) if all_ms else 1

        for name in s1:
            score = (s1[name]["accuracy"] * 0.4 +
                     s1[name]["f1"] * 0.4 +
                     (1 - s1[name]["inference_ms"] / max_ms) * 0.2)
            scores[name] = round(score, 4)

        best = max(scores, key=scores.get)
        print(f"  {'Model':<18} {'Edge Score':>12}")
        print("  " + "-" * 32)
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            marker = " ← RECOMMENDED" if name == best else ""
            print(f"  {name:<18} {score:>12.4f}{marker}")

    print(f"\n{'='*75}")
    print(f"  All outputs saved in: {out_dir}")
    print(f"  Use these numbers directly in your paper's Results section!\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="WoundAI Multi-Model Comparison — GPU/CPU"
    )
    parser.add_argument("--data_dir",   default="./datasets")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--skip_s1",    action="store_true")
    parser.add_argument("--skip_s2",    action="store_true")
    parser.add_argument("--skip_s3",    action="store_true")
    parser.add_argument("--no_mixed_precision", action="store_true",
                        help="Disable float16 mixed precision even on CUDA GPU")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Force CPU-only mode (useful for debugging)")
    args = parser.parse_args()

    # ── GPU / precision setup ─────────────────────────────────────────────────
    gpu_available = setup_gpu(force_cpu=args.no_gpu)
    if args.no_mixed_precision and gpu_available:
        tf.keras.mixed_precision.set_global_policy("float32")
        print("  ℹ️   Mixed precision disabled (--no_mixed_precision)")

    # Increase batch size automatically when GPU is available
    global BATCH_SIZE
    if gpu_available:
        BATCH_SIZE = max(BATCH_SIZE, 32)
        print(f"  📦  Batch size set to {BATCH_SIZE} (GPU mode)")

    datasets_dir = Path(args.data_dir)
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_gpu = len(tf.config.list_physical_devices("GPU"))
    device_str = f"GPU ×{n_gpu}" if gpu_available else "CPU"

    print("=" * 65)
    print("  WoundAI — Multi-Model Comparison Pipeline")
    print("  Models: CustomCNN | MobileNetV2 | ResNet50V2 |")
    print("          EfficientNetB0 | EfficientNetB3")
    print("=" * 65)
    print(f"  Data   : {datasets_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Device : {device_str}")
    print(f"  Batch  : {BATCH_SIZE}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    timeline = Timeline("WoundAI Training Pipeline")

    if not datasets_dir.exists():
        print(f"\n❌ Dataset folder not found: {datasets_dir}")
        print("   Run prepare_datasets.py first.\n")
        return

    s1_results, s2_results = {}, {}

    if not args.skip_s1:
        timeline.start("Stage 1 — Binary Wound Detector")
        s1_results = run_stage1(datasets_dir, out_dir)
        if s1_results:
            timeline.done("Stage 1 — Binary Wound Detector",
                          f"{len(s1_results)} models trained")
        else:
            timeline.fail("Stage 1 — Binary Wound Detector", "no results returned")
    else:
        timeline.skipped("Stage 1 — Binary Wound Detector", "--skip_s1")

    if not args.skip_s2:
        timeline.start("Stage 2 — Multi-class Classifier")
        s2_results = run_stage2(datasets_dir, out_dir)
        if s2_results:
            timeline.done("Stage 2 — Multi-class Classifier",
                          f"{len(s2_results)} models trained")
        else:
            timeline.fail("Stage 2 — Multi-class Classifier", "no results returned")
    else:
        timeline.skipped("Stage 2 — Multi-class Classifier", "--skip_s2")

    if not args.skip_s3:
        if s1_results and s2_results:
            timeline.start("Stage 3 — OOD Robustness")
            run_stage3(datasets_dir, out_dir, s1_results, s2_results)
            timeline.done("Stage 3 — OOD Robustness")
        else:
            timeline.skipped("Stage 3 — OOD Robustness",
                             "S1 and S2 results required")
            print("\n  ⚠️  Skipping Stage 3 — need Stage 1 and Stage 2 results first")
    else:
        timeline.skipped("Stage 3 — OOD Robustness", "--skip_s3")

    timeline.start("Final Comparison Table")
    print_final_table(out_dir)
    timeline.done("Final Comparison Table")

    timeline.save(out_dir / "timeline.json")
    timeline.print_summary()


if __name__ == "__main__":
    main()