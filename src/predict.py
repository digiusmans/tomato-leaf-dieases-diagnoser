from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_tomato_model.keras"
LABELS_PATH = MODELS_DIR / "labels.txt"
CONFIG_PATH = MODELS_DIR / "model_config.json"

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)

MIN_IMAGE_DIMENSION = 100
MIN_BRIGHTNESS = 20
MAX_BRIGHTNESS = 245
MIN_COLOR_STD = 25
MIN_GREEN_RATIO = 0.20
MIN_VALID_CONFIDENCE = 75.0
MIN_CONFIDENCE_GAP = 20.0


DISPLAY_NAME_MAP: Dict[str, str] = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___healthy": "Healthy",
}

DISPLAY_TO_RAW_NAME_MAP: Dict[str, str] = {display: raw for raw, display in DISPLAY_NAME_MAP.items()}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _load_labels(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    with path.open("r", encoding="utf-8") as file_handle:
        labels = [line.strip().strip('"').strip("'") for line in file_handle if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def _parse_size_value(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None

    if isinstance(value, dict):
        height = value.get("height") or value.get("h")
        width = value.get("width") or value.get("w")
        if height and width:
            return int(width), int(height)

    if isinstance(value, (list, tuple)):
        numeric_values = [int(item) for item in value if isinstance(item, (int, float))]
        if len(numeric_values) >= 2:
            if len(numeric_values) >= 3 and numeric_values[2] in (1, 3):
                return int(numeric_values[1]), int(numeric_values[0])
            return int(numeric_values[1]), int(numeric_values[0])

    if isinstance(value, str):
        parts = [part for part in re.split(r"[xX, ]+", value.strip()) if part]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[1]), int(parts[0])

    return None


def _extract_image_size(config: Dict[str, Any]) -> Tuple[int, int]:
    preferred_keys = [
        "image_size",
        "input_size",
        "target_size",
        "img_size",
        "input_shape",
        "image_shape",
        "model_input_shape",
    ]

    for key in preferred_keys:
        parsed_size = _parse_size_value(config.get(key))
        if parsed_size:
            return parsed_size

    height = config.get("img_height") or config.get("height")
    width = config.get("img_width") or config.get("width")
    if height and width:
        return int(width), int(height)

    return DEFAULT_IMAGE_SIZE


def _load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def _resolve_display_name(label: str) -> str:
    if label in DISPLAY_TO_RAW_NAME_MAP:
        return label
    return DISPLAY_NAME_MAP.get(label, label)


def _ensure_probability_distribution(raw_output: np.ndarray) -> np.ndarray:
    raw_output = np.asarray(raw_output, dtype=np.float32).reshape(-1)

    if raw_output.size == 0:
        raise ValueError("Model returned an empty prediction array.")

    if np.all(raw_output >= 0.0) and np.all(raw_output <= 1.0) and np.isclose(float(raw_output.sum()), 1.0, atol=1e-3):
        probabilities = raw_output
    else:
        probabilities = tf.nn.softmax(raw_output).numpy().astype(np.float32)

    return probabilities


MODEL_CONFIG = _load_json(CONFIG_PATH)
IMAGE_SIZE = _extract_image_size(MODEL_CONFIG)
MODEL = _load_model()
LABELS = _load_labels(LABELS_PATH)

DISEASE_INFO: Dict[str, Dict[str, str]] = {
    "Bacterial Spot": {
        "name": "Bacterial Spot",
        "description": "A bacterial disease that causes leaf and fruit spots, especially in warm, wet conditions.",
        "symptoms": "Dark, water-soaked spots on leaves, stems, and fruit; leaves may yellow and drop early.",
        "prevention": "Use certified seed, rotate crops, avoid overhead watering, and sanitize tools and plant debris.",
        "treatment": "Remove heavily infected tissue and apply copper-based bactericides early when needed.",
    },
    "Early Blight": {
        "name": "Early Blight",
        "description": "A common fungal disease that starts on lower leaves and spreads upward in humid weather.",
        "symptoms": "Brown spots with concentric rings, yellowing leaves, and defoliation starting from the bottom.",
        "prevention": "Rotate crops, mulch soil, stake plants, and keep leaves dry during irrigation.",
        "treatment": "Remove infected leaves and use approved fungicides such as copper or chlorothalonil as directed.",
    },
    "Late Blight": {
        "name": "Late Blight",
        "description": "A fast-spreading disease that can destroy tomato foliage and fruit under cool, wet conditions.",
        "symptoms": "Large dark lesions, pale green edges, white fuzz on leaf undersides, and rapidly collapsing plants.",
        "prevention": "Destroy volunteers, improve airflow, avoid wet foliage, and monitor closely during rainy periods.",
        "treatment": "Remove infected plants promptly and apply fungicides immediately when disease pressure is high.",
    },
    "Leaf Mold": {
        "name": "Leaf Mold",
        "description": "A fungal disease that is most common in greenhouses and humid, poorly ventilated spaces.",
        "symptoms": "Yellow spots on upper leaf surfaces and olive-green mold on the underside of leaves.",
        "prevention": "Improve ventilation, reduce humidity, space plants well, and avoid watering foliage.",
        "treatment": "Remove infected leaves and use fungicides if symptoms continue to spread.",
    },
    "Septoria Leaf Spot": {
        "name": "Septoria Leaf Spot",
        "description": "A fungal leaf spot disease that often appears after fruit set and lowers plant vigor.",
        "symptoms": "Small circular spots with dark margins and gray centers, usually beginning on lower leaves.",
        "prevention": "Practice crop rotation, clean up debris, mulch soil, and avoid splashing water onto leaves.",
        "treatment": "Remove infected lower leaves and apply labeled fungicides if the disease is advancing.",
    },
    "Spider Mites": {
        "name": "Spider Mites",
        "description": "Tiny sap-feeding pests that weaken plants and cause stippling, bronzing, and webbing.",
        "symptoms": "Fine yellow stippling, dusty or bronzed leaves, webbing, and leaf drop in severe infestations.",
        "prevention": "Inspect leaf undersides, reduce plant stress, control dust, and encourage beneficial insects.",
        "treatment": "Wash plants, use insecticidal soap or neem, and apply miticides when infestations are severe.",
    },
    "Target Spot": {
        "name": "Target Spot",
        "description": "A fungal disease that creates target-like lesions and can cause early leaf drop.",
        "symptoms": "Circular brown spots with rings, yellow halos, and progressive defoliation.",
        "prevention": "Rotate crops, remove crop residue, improve airflow, and avoid wetting foliage.",
        "treatment": "Remove infected tissue and use fungicides labeled for target spot management.",
    },
    "Yellow Leaf Curl Virus": {
        "name": "Yellow Leaf Curl Virus",
        "description": "A viral disease transmitted mainly by whiteflies that can severely stunt tomato plants.",
        "symptoms": "Upward leaf curl, strong yellowing, thickened leaves, stunting, and poor fruit set.",
        "prevention": "Control whiteflies, use resistant varieties, install insect netting, and remove infected plants early.",
        "treatment": "There is no cure; remove infected plants and focus on preventing spread by whiteflies.",
    },
    "Tomato Mosaic Virus": {
        "name": "Tomato Mosaic Virus",
        "description": "A highly contagious virus that spreads through contact, tools, and contaminated hands or plant material.",
        "symptoms": "Mosaic mottling, distorted leaves, reduced growth, and uneven fruit ripening.",
        "prevention": "Disinfect tools, wash hands, avoid tobacco handling around plants, and use resistant varieties.",
        "treatment": "There is no cure; remove infected plants and prevent mechanical spread to healthy plants.",
    },
    "Healthy": {
        "name": "Healthy",
        "description": "The plant appears free of visible disease symptoms at the time of inspection.",
        "symptoms": "Normal leaf color and shape, steady growth, and no obvious spotting, curling, or mold.",
        "prevention": "Keep up regular scouting, balanced nutrition, proper watering, and good airflow.",
        "treatment": "No treatment needed; continue routine monitoring and good crop management.",
    },
}


def preprocess_image(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("preprocess_image expects a PIL Image input.")

    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(resized_image, dtype=np.float32)
    batched_array = np.expand_dims(image_array, axis=0)
    return batched_array


def validate_input_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Validate whether uploaded image looks like a plausible tomato leaf image.
    """

    if not isinstance(image, Image.Image):
        return False, "Invalid image format."

    width, height = image.size

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return False, "Image is too small. Please upload a clearer tomato leaf image."

    rgb_image = image.convert("RGB")
    image_array = np.asarray(rgb_image, dtype=np.float32)

    if image_array.size == 0:
        return False, "Empty or corrupted image."

    avg_brightness = np.mean(image_array)

    if avg_brightness < MIN_BRIGHTNESS:
        return False, "Image is too dark. Please upload a brighter image."

    if avg_brightness > MAX_BRIGHTNESS:
        return False, "Image is too bright or washed out."

    color_std = np.std(image_array)

    if color_std < MIN_COLOR_STD:
        return False, "Image lacks enough visual detail."

    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]

    green_dominant = np.logical_and(green > red, green > blue)
    green_ratio = np.mean(green_dominant)

    if green_ratio < MIN_GREEN_RATIO:
        return False, "Uploaded image does not appear to contain a tomato leaf."

    return True, ""


def predict_disease(image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
    """
    Predict tomato disease with validation and top-k results.
    """

    is_valid, error_message = validate_input_image(image)

    if not is_valid:
        return {
            "valid": False,
            "error": error_message,
            "predictions": []
        }

    image_batch = preprocess_image(image)

    raw_predictions = MODEL.predict(image_batch, verbose=0)

    if isinstance(raw_predictions, (list, tuple)):
        raw_predictions = raw_predictions[0]

    probabilities = _ensure_probability_distribution(np.asarray(raw_predictions))

    class_count = min(len(probabilities), len(LABELS))

    if class_count == 0:
        return {
            "valid": False,
            "error": "Prediction failed. No valid class outputs found.",
            "predictions": []
        }

    probabilities = probabilities[:class_count]

    top_count = min(top_k, class_count)
    top_indices = np.argsort(probabilities)[::-1][:top_count]

    results = []

    for class_index in top_indices:
        raw_label = LABELS[class_index]
        display_label = DISPLAY_NAME_MAP.get(raw_label, raw_label)
        confidence = round(float(probabilities[class_index] * 100.0), 2)

        results.append({
            "label": display_label,
            "confidence": confidence
        })

    top_confidence = results[0]["confidence"]
    top2_gap = results[0]["confidence"] - results[1]["confidence"]

    if top_confidence < MIN_VALID_CONFIDENCE or top2_gap < MIN_CONFIDENCE_GAP:
        return {
            "valid": False,
            "error": "This image does not clearly appear to be a valid tomato leaf disease sample.",
            "predictions": results
        }

    return {
        "valid": True,
        "error": None,
        "predictions": results
    }


def get_confidence_level(confidence: float) -> str:
    if confidence >= 85:
        return "High Confidence"
    if confidence >= 60:
        return "Moderate Confidence"
    return "Low Confidence"


def get_disease_info(disease_name: str) -> Dict[str, str]:
    normalized_name = _resolve_display_name(disease_name)
    info = DISEASE_INFO.get(normalized_name)

    if info is not None:
        return dict(info)

    return {
        "name": normalized_name or "Unknown",
        "description": "No disease information is available for this label.",
        "symptoms": "Unknown",
        "prevention": "Unknown",
        "treatment": "Unknown",
    }