# Multistage Wound Screening AI

A three-stage deep learning pipeline for clinically safe wound screening: binary wound detection, multi-class wound type classification, and out-of-distribution (OOD) robustness evaluation for emergency and dermatology settings.

## Why Multi-Stage

Single-model wound classifiers fail silently on inputs they weren't trained for, an image of a burn misclassified with false confidence, or a non-wound image processed as if it were one. This pipeline separates the problem into three stages so each failure mode is caught explicitly rather than masked by a single softmax output.

## Pipeline

**Stage 1: Binary Wound Detection** — Filters non-wound images before they reach classification. Trained and benchmarked 5 architectures (CustomCNN, MobileNetV2, ResNet50V2, EfficientNetB0, EfficientNetB3) on 6,705 images (2,788 wound / 3,917 non-wound). Best: MobileNetV2 at 96.3% accuracy, 0.994 AUC.

**Stage 2: Multi-Class Wound Type Classification** — Classifies detected wounds into 9 clinical categories (bruise, pressure ulcer, infected wound, abrasion, laceration, venous ulcer, burn, cut, diabetic ulcer) across 746 images, evaluated with 5-fold cross-validation. Best: EfficientNetB3 at 71.3% CV accuracy (±9.8%), 0.630 CV F1.

**Stage 3: OOD Robustness Evaluation** — Tests all 5 models against 233 out-of-distribution images (miscellaneous, orthopedic, and abdominal wounds outside the training distribution) to measure false-positive rate on inputs the system should flag as uncertain rather than classify confidently.

**Edge Deployment Scoring** — Composite score across accuracy, inference speed, and model size to select a deployment target. MobileNetV2 scored highest (0.908) and was exported to TFLite for mobile/edge use.

## Results

**Stage 1 (Binary Detection)**

| Model | Acc | F1 | AUC | Inference |
|---|---|---|---|---|
| MobileNetV2 | 0.963 | 0.954 | 0.994 | 4.5ms |
| EfficientNetB0 | 0.963 | 0.954 | 0.995 | 7.6ms |
| EfficientNetB3 | 0.963 | 0.954 | 0.994 | 11.3ms |
| ResNet50V2 | 0.949 | 0.938 | 0.987 | 5.0ms |
| CustomCNN | 0.907 | 0.876 | 0.982 | 2.1ms |

**Stage 2 (Multi-Class, 5-fold CV)**

| Model | CV Acc | CV F1 |
|---|---|---|
| EfficientNetB3 | 0.713 ± 0.098 | 0.630 ± 0.102 |
| EfficientNetB0 | 0.625 ± 0.106 | 0.555 ± 0.108 |
| MobileNetV2 | 0.610 ± 0.166 | 0.520 ± 0.189 |
| ResNet50V2 | 0.548 ± 0.146 | 0.470 ± 0.161 |
| CustomCNN | 0.239 ± 0.022 | 0.090 ± 0.025 |

## Known Limitation

Stage 3 OOD detection did not reach the target false-positive rate (<10%) on any architecture, the best (CustomCNN) came in at 66.5% FPR. This means the system currently over-trusts out-of-distribution inputs more than a clinical deployment would require. Documented honestly here as the clearest next research direction, likely candidates: confidence calibration, a dedicated OOD detection head, or Mahalanobis-distance-based rejection.

## Tech Stack

TensorFlow/Keras, Grad-CAM for interpretability (`gradcam.py`), TFLite for edge export, trained via DirectML (no CUDA GPU required, runs on integrated/AMD graphics via `Train_cpu.py`).

## Setup

```bash
pip install -r Requirements.txt
python prepare_dataset.py
python run_pipeline.py
```

Or on Windows: `run_training.bat`

Run the app locally: `python app.py`

## Contributors

Built with [Urva Ishfaq](https://github.com/urva-ishfaq393).
