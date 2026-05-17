from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "tflite"


MODELS = {
    "stage1_customcnn": RESULTS_DIR / "stage1" / "CustomCNN" / "best.keras",
    "stage1_mobilenetv2": RESULTS_DIR / "stage1" / "MobileNetV2" / "best.keras",
    "stage1_efficientnetb0": RESULTS_DIR / "stage1" / "EfficientNetB0" / "best.keras",
    "stage1_efficientnetb3": RESULTS_DIR / "stage1" / "EfficientNetB3" / "best.keras",
    "stage1_resnet50v2": RESULTS_DIR / "stage1" / "ResNet50V2" / "best.keras",
    "stage2_customcnn": RESULTS_DIR / "stage2" / "CustomCNN" / "best.keras",
    "stage2_mobilenetv2": RESULTS_DIR / "stage2" / "MobileNetV2" / "best.keras",
    "stage2_efficientnetb0": RESULTS_DIR / "stage2" / "EfficientNetB0" / "best.keras",
    "stage2_efficientnetb3": RESULTS_DIR / "stage2" / "EfficientNetB3" / "best.keras",
    "stage2_resnet50v2": RESULTS_DIR / "stage2" / "ResNet50V2" / "best.keras",
}


def convert_model(name: str, model_path: Path) -> None:
    if not model_path.exists():
        print(f"SKIP {name}: missing {model_path}")
        return

    print(f"Loading {name}: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{name}.tflite"
    output_path.write_bytes(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {output_path} ({size_mb:.2f} MB)")


def main() -> None:
    print(f"TensorFlow: {tf.__version__}")
    print(f"Output folder: {OUT_DIR}")

    for name, model_path in MODELS.items():
        try:
            convert_model(name, model_path)
        except Exception as exc:
            print(f"FAILED {name}: {exc}")


if __name__ == "__main__":
    main()
