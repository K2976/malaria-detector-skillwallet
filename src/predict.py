"""
predict.py
===========
Single-image inference module for malaria detection.

Public API
----------
``predict_image(image_path)`` → dict with ``label`` and ``confidence``.

Usage
-----
    python src/predict.py path/to/cell_image.png
"""

import os
import sys

# Ensure Keras 2 legacy behavior is used if using TensorFlow >= 2.16
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_SIZE = (130, 130)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "malaria_detector.h5")

# Class labels — these must match the order used during training.
# flow_from_directory sorts alphabetically:  Parasitized=0, Uninfected=1
CLASS_LABELS = {0: "Parasitized", 1: "Uninfected"}

# Cache the model so repeated calls don't reload from disk.
_model_cache = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load and cache the Keras model."""
    if model_path not in _model_cache:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Train the model first using `python src/train_model.py`."
            )
        print(f"📂 Loading model from {model_path} …")
        _model_cache[model_path] = load_model(model_path)
    return _model_cache[model_path]


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Read an image from disk, resize to 224×224, normalise to [0, 1] and
    return a batch-ready array of shape ``(1, 224, 224, 3)``.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_image(image_path: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Run malaria detection on a single image.

    Parameters
    ----------
    image_path : str
        Path to the microscopic blood-smear cell image.
    model_path : str, optional
        Path to the saved ``.h5`` model file.

    Returns
    -------
    dict
        ``{"label": "Parasitized" | "Uninfected", "confidence": float}``
    """
    model = _load_model(model_path)
    img = preprocess_image(image_path)

    prediction = model.predict(img, verbose=0)[0][0]

    # Sigmoid output: values close to 0 → Parasitized, close to 1 → Uninfected
    if prediction >= 0.5:
        label = "Uninfected"
        confidence = float(prediction)
    else:
        label = "Parasitized"
        confidence = float(1 - prediction)

    return {"label": label, "confidence": round(confidence * 100, 2)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    result = predict_image(path)
    print(f"\n🔬 Prediction : {result['label']}")
    print(f"   Confidence : {result['confidence']:.2f}%\n")
