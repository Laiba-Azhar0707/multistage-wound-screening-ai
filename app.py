"""
WoundAI Backend API
====================
FastAPI server that:
  1. Accepts wound image uploads
  2. Stage 1 → Detects if it's a wound
  3. Stage 2 → Classifies wound type
  4. Stage 3 → Analyzes boundary, infection signs, severity
  5. Returns full paramedic report as JSON

Install & Run:
    pip install fastapi uvicorn tensorflow pillow opencv-python numpy python-multipart
    python app.py

API will run at: http://localhost:8000
Swagger docs at: http://localhost:8000/docs
"""

import os
import io
import cv2
import json
import time
import base64
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE     = 96
RESULTS_DIR  = Path("H:/Wound_Screening_app/results")
MODELS_DIR   = RESULTS_DIR

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("WoundAI")

# ─── WOUND TYPE DESCRIPTIONS ──────────────────────────────────────────────────
WOUND_DESCRIPTIONS = {
    "abrasion": {
        "description": "Superficial wound where skin is scraped or rubbed away.",
        "infection_risk": "Low-Medium",
        "first_aid": [
            "Clean the wound with clean water and mild soap",
            "Remove any visible debris gently",
            "Apply antiseptic solution",
            "Cover with sterile non-stick dressing",
            "Change dressing daily"
        ],
        "watch_for": ["Redness spreading beyond wound", "Pus or discharge", "Increased warmth"]
    },
    "bruise": {
        "description": "Bleeding under the skin without breaking the surface, caused by blunt trauma.",
        "infection_risk": "Low",
        "first_aid": [
            "Apply ice pack wrapped in cloth for 15-20 min",
            "Elevate the affected area if possible",
            "Rest the injured area",
            "Apply compression bandage if swelling is significant"
        ],
        "watch_for": ["Severe swelling", "Inability to move limb", "Bruise not fading after 2 weeks"]
    },
    "burn": {
        "description": "Tissue damage from heat, chemicals, electricity, or radiation.",
        "infection_risk": "High",
        "first_aid": [
            "Cool the burn under running cool water for 20 minutes",
            "Do NOT use ice, butter, or toothpaste",
            "Remove jewelry/clothing near burn (if not stuck)",
            "Cover loosely with sterile non-fluffy material",
            "Seek immediate medical attention for burns > 3cm"
        ],
        "watch_for": ["Blistering", "Charred or white skin (3rd degree)", "Burn around face/hands/genitals"]
    },
    "cut": {
        "description": "Clean incision through the skin, typically with a sharp object.",
        "infection_risk": "Medium",
        "first_aid": [
            "Apply direct pressure with clean cloth to stop bleeding",
            "Once bleeding stops, clean with clean water",
            "Apply antiseptic",
            "Close wound edges with butterfly strips if gaping",
            "Cover with sterile dressing"
        ],
        "watch_for": ["Bleeding not stopping after 10 min", "Deep wound exposing fat/muscle", "Signs of infection after 24h"]
    },
    "diabetic_wound": {
        "description": "Chronic wound common in diabetics, often on feet, with poor healing capacity.",
        "infection_risk": "Very High",
        "first_aid": [
            "Do NOT attempt to treat at home — refer to medical professional immediately",
            "Keep area clean and dry",
            "Do not apply pressure to the wound",
            "Document wound size and appearance for medical team"
        ],
        "watch_for": ["Blackening/necrosis", "Strong odor (gangrene)", "Fever", "Red streaks from wound"]
    },
    "laceration": {
        "description": "Irregular, jagged wound caused by blunt force or tearing.",
        "infection_risk": "High",
        "first_aid": [
            "Control bleeding with direct pressure",
            "Do NOT remove embedded objects",
            "Irrigate wound thoroughly with clean water",
            "Cover with sterile dressing",
            "Transport to hospital — may require sutures"
        ],
        "watch_for": ["Arterial bleeding (bright red, spurting)", "Deep penetrating wound", "Loss of sensation"]
    },
    "normal": {
        "description": "Healthy skin with no wound detected.",
        "infection_risk": "None",
        "first_aid": ["No treatment required"],
        "watch_for": []
    },
    "pressure_wound": {
        "description": "Skin damage from sustained pressure, common in bedridden patients.",
        "infection_risk": "Very High",
        "first_aid": [
            "Relieve pressure from the area immediately",
            "Reposition patient every 2 hours",
            "Keep area clean and moisturized",
            "Use specialized pressure-relief mattress/cushion",
            "Refer to wound care specialist"
        ],
        "watch_for": ["Stage 3/4 deep tissue damage", "Signs of sepsis", "Tunneling wounds"]
    },
    "surgical_wound": {
        "description": "Post-operative incision requiring careful monitoring for healing.",
        "infection_risk": "Medium",
        "first_aid": [
            "Keep wound dry for 24-48 hours post-surgery",
            "Change dressing as instructed by surgeon",
            "Watch for dehiscence (opening of wound)",
            "Keep follow-up appointments"
        ],
        "watch_for": ["Wound opening", "Discharge other than clear fluid", "Fever > 38°C", "Increased pain after day 3"]
    }
}

SEVERITY_RULES = {
    "normal":          (0,  "None",     "No action required"),
    "bruise":          (2,  "Mild",     "Monitor at home"),
    "abrasion":        (3,  "Mild",     "Clean and dress the wound"),
    "cut":             (5,  "Moderate", "Medical assessment recommended"),
    "laceration":      (7,  "Serious",  "Urgent medical care needed"),
    "burn":            (8,  "Serious",  "Seek immediate medical attention"),
    "surgical_wound":  (6,  "Moderate", "Follow post-operative care plan"),
    "pressure_wound":  (8,  "Serious",  "Refer to wound care specialist"),
    "diabetic_wound":  (9,  "Critical", "IMMEDIATE medical attention required"),
}

# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.s1_model = None
        self.s2_model = None
        self.s1_class_names = []
        self.s2_class_names = []
        self._load()

    def _best_model_path(self, stage: str) -> Optional[Path]:
        """
        Auto-selects best model based on saved results.
        Falls back to any available model.
        """
        results_file = MODELS_DIR / stage / "all_results.json"
        if results_file.exists():
            data = json.load(open(results_file))
            models = data.get("models", {})
            if models:
                # Pick model with highest accuracy
                best = max(models, key=lambda m: models[m].get("accuracy", 0))
                best_path = MODELS_DIR / stage / best / "best.keras"
                if best_path.exists():
                    log.info(f"  Auto-selected {stage}: {best}")
                    return best_path, data.get("class_names", [])

        # Fallback: scan for any saved model
        for model_name in ["CustomCNN", "MobileNetV2", "EfficientNetB0",
                            "EfficientNetB3", "ResNet50V2"]:
            p = MODELS_DIR / stage / model_name / "best.keras"
            if p.exists():
                log.info(f"  Fallback {stage}: {model_name}")
                return p, []

        return None, []

    def _load(self):
        log.info("Loading models...")
        s1_path, s1_classes = self._best_model_path("stage1")
        s2_path, s2_classes = self._best_model_path("stage2")

        if s1_path:
            self.s1_model = tf.keras.models.load_model(str(s1_path))
            self.s1_class_names = s1_classes or ["non_wound", "wound"]
            log.info(f"✅ Stage 1 loaded: {s1_path.parent.name}")
        else:
            log.warning("⚠️  Stage 1 model not found — running in demo mode")

        if s2_path:
            self.s2_model = tf.keras.models.load_model(str(s2_path))
            self.s2_class_names = s2_classes or list(WOUND_DESCRIPTIONS.keys())
            log.info(f"✅ Stage 2 loaded: {s2_path.parent.name}")
        else:
            log.warning("⚠️  Stage 2 model not found — running in demo mode")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32)
        return np.expand_dims(img, axis=0)

    def predict_stage1(self, img_tensor):
        if self.s1_model is None:
            return 0.85, "wound"  # demo mode
        prob = float(self.s1_model.predict(img_tensor, verbose=0)[0][0])
        label = self.s1_class_names[int(prob >= 0.5)] if self.s1_class_names else (
            "wound" if prob >= 0.5 else "non_wound")
        return prob, label

    def predict_stage2(self, img_tensor):
        if self.s2_model is None:
            return "cut", 0.75, {"cut": 0.75, "laceration": 0.15, "abrasion": 0.10}
        probs = self.s2_model.predict(img_tensor, verbose=0)[0]
        idx   = int(np.argmax(probs))
        label = self.s2_class_names[idx] if idx < len(self.s2_class_names) else "unknown"
        conf  = float(probs[idx])
        top3  = {self.s2_class_names[i]: float(probs[i])
                 for i in np.argsort(probs)[::-1][:3]
                 if i < len(self.s2_class_names)}
        return label, conf, top3


# ─── WOUND ANALYSIS ───────────────────────────────────────────────────────────
class WoundAnalyzer:
    """Computer vision analysis of wound boundary, color, and infection signs."""

    def analyze(self, image: np.ndarray, wound_type: str) -> dict:
        result = {}
        result["boundary"]  = self._boundary_analysis(image)
        result["color"]     = self._color_analysis(image)
        result["infection"] = self._infection_signs(image, wound_type)
        result["size_estimate"] = self._size_estimate(image)
        return result

    def _boundary_analysis(self, img: np.ndarray) -> dict:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"regularity": "Indeterminate", "sharpness": "Indeterminate",
                    "edge_density": 0.0}

        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        perim   = cv2.arcLength(largest, True)

        # Circularity — higher = more regular boundary
        circularity = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0
        regularity  = ("Regular" if circularity > 0.6 else
                       "Irregular" if circularity > 0.3 else "Very Irregular")

        edge_density = float(np.sum(edges > 0) / edges.size)
        sharpness    = "Sharp" if edge_density > 0.05 else "Diffuse"

        return {
            "regularity": regularity,
            "sharpness": sharpness,
            "edge_density": round(edge_density, 4),
            "circularity": round(float(circularity), 3)
        }

    def _color_analysis(self, img: np.ndarray) -> dict:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        # Detect dominant color regions
        red_mask    = ((h < 15) | (h > 165)) & (s > 50)
        yellow_mask = (h > 20) & (h < 35) & (s > 80)
        black_mask  = v < 40
        white_mask  = (v > 200) & (s < 40)
        green_mask  = (h > 35) & (h < 85) & (s > 50)

        total = img.shape[0] * img.shape[1]

        return {
            "red_percent":    round(float(red_mask.sum()    / total * 100), 1),
            "yellow_percent": round(float(yellow_mask.sum() / total * 100), 1),
            "black_percent":  round(float(black_mask.sum()  / total * 100), 1),
            "white_percent":  round(float(white_mask.sum()  / total * 100), 1),
            "green_percent":  round(float(green_mask.sum()  / total * 100), 1),
        }

    def _infection_signs(self, img: np.ndarray, wound_type: str) -> dict:
        colors = self._color_analysis(img)
        signs  = []

        # Yellow/green = pus, possible bacterial infection
        if colors["yellow_percent"] > 8:
            signs.append("Possible purulent discharge (pus) — bacterial infection suspected")
        if colors["green_percent"] > 5:
            signs.append("Greenish discoloration — Pseudomonas or fungal infection possible")
        if colors["black_percent"] > 10:
            signs.append("Necrotic tissue (blackening) — urgent debridement may be needed")
        if colors["red_percent"] > 40:
            signs.append("Significant erythema — active inflammation or cellulitis")
        if colors["white_percent"] > 15:
            signs.append("White coating — possible fungal colonization (candida)")

        # Wound-type specific warnings
        if wound_type == "diabetic_wound":
            signs.append("Diabetic wound — high risk of polymicrobial infection")
        if wound_type == "pressure_wound":
            signs.append("Pressure injury — check for tunneling and undermining")

        bacterial_risk = "High" if len([s for s in signs if "bacterial" in s.lower() or "purulent" in s.lower()]) > 0 else \
                         "Medium" if len(signs) > 1 else "Low"
        fungal_risk    = "High" if any("fungal" in s.lower() or "candida" in s.lower() for s in signs) else "Low"

        return {
            "signs": signs if signs else ["No obvious infection signs detected"],
            "bacterial_risk": bacterial_risk,
            "fungal_risk": fungal_risk,
            "necrosis_present": colors["black_percent"] > 10
        }

    def _size_estimate(self, img: np.ndarray) -> dict:
        """
        Estimates relative wound coverage of the image frame.
        For accurate sizing, a reference object (coin) in frame is needed.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        wound_pixels = int(np.sum(thresh > 0))
        total_pixels = img.shape[0] * img.shape[1]
        coverage     = round(wound_pixels / total_pixels * 100, 1)

        size_cat = ("Small (<2cm)"    if coverage < 15 else
                    "Medium (2-5cm)"  if coverage < 40 else
                    "Large (>5cm)")

        return {
            "coverage_percent": coverage,
            "estimated_category": size_cat,
            "note": "Place a coin next to wound for accurate size measurement"
        }


# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
app     = FastAPI(title="WoundAI API", version="2.0")
manager = ModelManager()
analyzer = WoundAnalyzer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "WoundAI API running", "version": "2.0",
            "endpoints": ["/analyze", "/health", "/docs"]}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "stage1_loaded": manager.s1_model is not None,
        "stage2_loaded": manager.s2_model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/analyze")
async def analyze_wound(file: UploadFile = File(...)):
    t0 = time.time()

    # ── Read image ─────────────────────────────────────────────────────────
    if file.content_type not in ["image/jpeg", "image/png",
                                  "image/webp", "image/bmp"]:
        raise HTTPException(400, "Unsupported image format")

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(400, "Could not decode image")

    img_tensor = manager.preprocess(img_bgr)

    # ── Stage 1: Is it a wound? ────────────────────────────────────────────
    s1_prob, s1_label = manager.predict_stage1(img_tensor)
    is_wound = s1_label.lower() not in ["non_wound", "normal", "healthy"]

    if not is_wound:
        return JSONResponse({
            "is_wound": False,
            "confidence": round(1 - s1_prob, 3),
            "message": "No wound detected in this image.",
            "recommendation": "Image appears to show healthy skin. If you believe this is wrong, try a clearer, closer photo.",
            "processing_ms": round((time.time() - t0) * 1000, 1)
        })

    # ── Stage 2: What type of wound? ───────────────────────────────────────
    wound_type, s2_conf, top3 = manager.predict_stage2(img_tensor)

    # ── Stage 3: Visual analysis ───────────────────────────────────────────
    analysis = analyzer.analyze(img_bgr, wound_type)

    # ── Build report ───────────────────────────────────────────────────────
    wound_info   = WOUND_DESCRIPTIONS.get(wound_type, WOUND_DESCRIPTIONS["cut"])
    sev_score, sev_label, sev_action = SEVERITY_RULES.get(
        wound_type, (5, "Moderate", "Medical assessment recommended"))

    # Adjust severity based on infection signs
    if analysis["infection"]["necrosis_present"]:
        sev_score = min(10, sev_score + 2)
    if analysis["infection"]["bacterial_risk"] == "High":
        sev_score = min(10, sev_score + 1)
    if analysis["infection"]["fungal_risk"] == "High":
        sev_score = min(10, sev_score + 1)

    urgency = ("CRITICAL — Call Emergency Services" if sev_score >= 9 else
               "URGENT — Go to ER Now"             if sev_score >= 7 else
               "MODERATE — See a Doctor Today"     if sev_score >= 5 else
               "MILD — Manage at Home")

    report = {
        "is_wound": True,
        "timestamp": datetime.utcnow().isoformat(),

        # Classification
        "classification": {
            "wound_type": wound_type.replace("_", " ").title(),
            "wound_type_key": wound_type,
            "confidence": round(s2_conf, 3),
            "detection_confidence": round(s1_prob, 3),
            "top_predictions": {k.replace("_"," ").title(): round(v,3)
                                 for k, v in top3.items()}
        },

        # Description
        "description": {
            "summary": wound_info["description"],
            "infection_risk_base": wound_info["infection_risk"],
        },

        # Boundary analysis
        "boundary": {
            "regularity": analysis["boundary"]["regularity"],
            "edge_sharpness": analysis["boundary"]["sharpness"],
            "circularity_score": analysis["boundary"]["circularity"],
            "interpretation": (
                "Well-defined wound with clean margins" if analysis["boundary"]["regularity"] == "Regular"
                else "Irregular wound margins — may indicate blunt trauma or tearing"
                if analysis["boundary"]["regularity"] == "Irregular"
                else "Very irregular wound — possible severe laceration or bite wound"
            )
        },

        # Size
        "size": analysis["size_estimate"],

        # Infection analysis
        "infection_analysis": {
            "signs_detected": analysis["infection"]["signs"],
            "bacterial_risk": analysis["infection"]["bacterial_risk"],
            "fungal_risk": analysis["infection"]["fungal_risk"],
            "necrosis_present": analysis["infection"]["necrosis_present"],
            "color_profile": analysis["color"]
        },

        # Severity
        "severity": {
            "score": sev_score,
            "out_of": 10,
            "label": sev_label,
            "urgency": urgency,
            "recommended_action": sev_action
        },

        # First aid
        "first_aid": {
            "steps": wound_info["first_aid"],
            "warning_signs": wound_info["watch_for"],
            "do_not": _get_donots(wound_type)
        },

        "processing_ms": round((time.time() - t0) * 1000, 1)
    }

    return JSONResponse(report)


def _get_donots(wound_type: str) -> list:
    base = ["Do not touch wound with bare hands",
            "Do not use cotton wool directly on wound"]
    specific = {
        "burn":         ["Do NOT use ice or cold water after 20 min",
                         "Do NOT burst blisters",
                         "Do NOT apply butter or toothpaste"],
        "laceration":   ["Do NOT remove embedded objects",
                         "Do NOT probe the wound depth"],
        "diabetic_wound": ["Do NOT ignore even small wounds",
                            "Do NOT attempt home debridement"],
        "pressure_wound": ["Do NOT massage directly over wound",
                            "Do NOT use donut-shaped cushions"],
    }
    return base + specific.get(wound_type, [])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
