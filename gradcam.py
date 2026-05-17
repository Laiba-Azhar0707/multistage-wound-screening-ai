"""
WoundAI — Grad-CAM Heatmap Generator
======================================
Generates Grad-CAM heatmaps for wound classification explainability.

How it works:
  1. Hooks into the last convolutional layer of the model
  2. Computes gradients of the predicted class score w.r.t. that layer's output
  3. Weights the feature maps by their average gradient → produces a heatmap
  4. Overlays the heatmap on the original image (jet colormap)
  5. Returns the overlay as a base64-encoded JPEG string (ready for JSON/Flutter)

Usage in app.py:
    from gradcam import GradCAM
    gradcam = GradCAM()

    # In your /analyze endpoint, after stage2 prediction:
    heatmap_b64, cam_data = gradcam.generate(
        model      = manager.s2_model,
        img_bgr    = img_bgr,
        img_tensor = img_tensor,
        class_idx  = predicted_class_index   # from np.argmax(probs)
    )
    # Add to your report dict:
    report["gradcam"] = {
        "heatmap_base64": heatmap_b64,   # send to Flutter as image
        "focus_region":   cam_data["focus_region"],
        "attention_score": cam_data["attention_score"],
        "interpretation": cam_data["interpretation"]
    }
"""

import cv2
import numpy as np
import base64
import logging
import tensorflow as tf
from typing import Optional, Tuple

log = logging.getLogger("WoundAI.GradCAM")


# ─── LAST CONV LAYER FINDER ───────────────────────────────────────────────────

def find_last_conv_layer(model: tf.keras.Model) -> Optional[str]:
    """
    Automatically finds the last convolutional layer in any Keras model.
    Works with MobileNetV2, EfficientNetB0/B3, ResNet50V2, and CustomCNN.
    """
    last_conv = None
    for layer in model.layers:
        # Handle nested models (transfer learning base models)
        if isinstance(layer, tf.keras.Model):
            for inner_layer in layer.layers:
                if isinstance(inner_layer, (tf.keras.layers.Conv2D,
                                            tf.keras.layers.DepthwiseConv2D)):
                    last_conv = f"{layer.name}/{inner_layer.name}"
        elif isinstance(layer, (tf.keras.layers.Conv2D,
                                tf.keras.layers.DepthwiseConv2D)):
            last_conv = layer.name

    if last_conv:
        log.info(f"Grad-CAM target layer: {last_conv}")
    else:
        log.warning("No conv layer found — Grad-CAM will be unavailable")

    return last_conv


# ─── GRAD-CAM CORE ────────────────────────────────────────────────────────────

class GradCAM:
    """
    Generates Grad-CAM heatmaps for WoundAI models.
    Supports all 5 trained architectures automatically.
    """

    # Descriptions for where the model is focusing
    FOCUS_REGIONS = {
        "center":      "Model focused on the central wound area",
        "edges":       "Model focused on wound boundary/margins",
        "top_left":    "Model focused on upper-left region",
        "top_right":   "Model focused on upper-right region",
        "bottom_left": "Model focused on lower-left region",
        "bottom_right":"Model focused on lower-right region",
        "distributed": "Model attention distributed across multiple regions",
    }

    def generate(
        self,
        model:      tf.keras.Model,
        img_bgr:    np.ndarray,
        img_tensor: np.ndarray,
        class_idx:  int,
        alpha:      float = 0.5,
        colormap:   int   = cv2.COLORMAP_JET,
    ) -> Tuple[str, dict]:
        """
        Generate Grad-CAM heatmap overlay.

        Args:
            model:      Loaded Keras model (stage 1 or stage 2)
            img_bgr:    Original image in BGR format (H x W x 3)
            img_tensor: Preprocessed tensor (1 x IMG_SIZE x IMG_SIZE x 3)
            class_idx:  Index of the class to explain (from np.argmax)
            alpha:      Heatmap overlay transparency (0=invisible, 1=full)
            colormap:   OpenCV colormap (default: JET — blue→green→red)

        Returns:
            heatmap_b64: Base64-encoded JPEG string of overlay image
            cam_data:    Dict with focus_region, attention_score, interpretation
        """
        try:
            heatmap_raw = self._compute_gradcam(model, img_tensor, class_idx)
            overlay     = self._overlay_heatmap(img_bgr, heatmap_raw, alpha, colormap)
            heatmap_b64 = self._encode_base64(overlay)
            cam_data    = self._interpret_heatmap(heatmap_raw)

            return heatmap_b64, cam_data

        except Exception as e:
            log.error(f"Grad-CAM failed: {e}")
            # Return placeholder so app doesn't crash if Grad-CAM fails
            return "", {
                "focus_region":    "unavailable",
                "attention_score": 0.0,
                "interpretation":  f"Heatmap unavailable: {str(e)}"
            }

    # ── Private Methods ───────────────────────────────────────────────────────

    def _compute_gradcam(
        self,
        model:     tf.keras.Model,
        img_tensor: np.ndarray,
        class_idx: int
    ) -> np.ndarray:
        """
        Core Grad-CAM computation using GradientTape.
        Returns raw heatmap as float32 array (values 0–1).
        """
        conv_layer_name = find_last_conv_layer(model)
        if conv_layer_name is None:
            raise ValueError("No convolutional layer found in model")

        # Build sub-model: input → last conv layer outputs + final predictions
        # Handle nested layer names (e.g. "mobilenetv2/Conv_1")
        if "/" in conv_layer_name:
            base_name, inner_name = conv_layer_name.split("/", 1)
            base_model  = model.get_layer(base_name)
            conv_layer  = base_model.get_layer(inner_name)
        else:
            conv_layer = model.get_layer(conv_layer_name)

        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [conv_layer.output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            inputs        = tf.cast(img_tensor, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            # Score for target class
            loss = predictions[:, class_idx]

        # Gradient of class score w.r.t. conv feature maps
        grads = tape.gradient(loss, conv_outputs)

        # Pool gradients over spatial dimensions → importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by their importance
        conv_outputs = conv_outputs[0]  # Remove batch dim
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU: only positive activations matter for the target class
        heatmap = tf.maximum(heatmap, 0)

        # Normalize to 0–1
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    def _overlay_heatmap(
        self,
        img_bgr:     np.ndarray,
        heatmap_raw: np.ndarray,
        alpha:       float,
        colormap:    int
    ) -> np.ndarray:
        """
        Resizes heatmap to original image size and blends with the image.
        Returns BGR overlay image.
        """
        h, w = img_bgr.shape[:2]

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap_raw, (w, h))

        # Convert to uint8 and apply colormap
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # Blend: original image + heatmap
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay

    def _encode_base64(self, img_bgr: np.ndarray) -> str:
        """
        Encodes BGR image as base64 JPEG string.
        Flutter can decode this directly: Image.memory(base64Decode(heatmap_b64))
        """
        success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("Failed to encode heatmap image as JPEG")
        return base64.b64encode(buffer).decode("utf-8")

    def _interpret_heatmap(self, heatmap: np.ndarray) -> dict:
        """
        Analyzes WHERE the model is looking (focus region) and
        HOW confidently it focuses (attention score).
        """
        if heatmap.size == 0:
            return {"focus_region": "unknown", "attention_score": 0.0,
                    "interpretation": "Could not analyze heatmap"}

        h, w = heatmap.shape

        # Attention score = mean of top 20% activations (how focused the model is)
        top_threshold  = np.percentile(heatmap, 80)
        top_activations = heatmap[heatmap >= top_threshold]
        attention_score = float(np.mean(top_activations)) if len(top_activations) > 0 else 0.0

        # Find center of mass of high-activation region
        y_coords, x_coords = np.where(heatmap >= top_threshold)
        if len(y_coords) == 0:
            return {"focus_region": "distributed", "attention_score": round(attention_score, 3),
                    "interpretation": self.FOCUS_REGIONS["distributed"]}

        cy = np.mean(y_coords) / h  # normalized 0–1
        cx = np.mean(x_coords) / w  # normalized 0–1

        # Determine focus region
        center_zone = 0.25  # 25% margin from center counts as "center"
        if abs(cx - 0.5) < center_zone and abs(cy - 0.5) < center_zone:
            region = "center"
        elif cx < 0.5 and cy < 0.5:
            region = "top_left"
        elif cx >= 0.5 and cy < 0.5:
            region = "top_right"
        elif cx < 0.5 and cy >= 0.5:
            region = "bottom_left"
        else:
            region = "bottom_right"

        # Check if attention is spread (low variance = distributed)
        activation_std = float(np.std(heatmap))
        if activation_std < 0.1:
            region = "distributed"

        # Clinical interpretation based on region + attention score
        if attention_score > 0.7:
            confidence_text = "High model confidence"
        elif attention_score > 0.4:
            confidence_text = "Moderate model confidence"
        else:
            confidence_text = "Low model confidence — consider retaking photo"

        interpretation = f"{self.FOCUS_REGIONS[region]}. {confidence_text}."

        return {
            "focus_region":    region,
            "attention_score": round(attention_score, 3),
            "center_of_mass":  {"x": round(float(cx), 3), "y": round(float(cy), 3)},
            "interpretation":  interpretation
        }


# ─── STANDALONE TEST ──────────────────────────────────────────────────────────
# Run this file directly to test Grad-CAM on a sample image:
#   python gradcam.py --model results/stage2/EfficientNetB3/best.keras --image test.jpg

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test Grad-CAM on a single image")
    parser.add_argument("--model", required=True, help="Path to best.keras model")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--class_idx", type=int, default=0, help="Class index to explain")
    parser.add_argument("--output", default="gradcam_output.jpg", help="Output overlay path")
    args = parser.parse_args()

    IMG_SIZE = 96

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    print(f"Loading image: {args.image}")
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print("ERROR: Could not read image")
        exit(1)

    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    img_tensor  = np.expand_dims(img_resized, axis=0)

    print(f"Generating Grad-CAM for class index: {args.class_idx}")
    gcam = GradCAM()
    heatmap_b64, cam_data = gcam.generate(model, img_bgr, img_tensor, args.class_idx)

    # Save overlay to disk
    overlay_bytes = base64.b64decode(heatmap_b64)
    with open(args.output, "wb") as f:
        f.write(overlay_bytes)

    print(f"\nGrad-CAM Results:")
    print(f"  Focus region   : {cam_data['focus_region']}")
    print(f"  Attention score: {cam_data['attention_score']}")
    print(f"  Interpretation : {cam_data['interpretation']}")
    print(f"\nOverlay saved to: {args.output}")