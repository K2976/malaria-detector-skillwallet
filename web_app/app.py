"""
app.py
======
Flask web application for malaria detection.

Routes
------
/         — Upload page (GET)
/predict  — Process uploaded image and display results (POST)
"""

import os
import sys
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Add project src/ to path so we can import predict module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from predict import predict_image  # noqa: E402

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder=os.path.join(PROJECT_ROOT, "static"),
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)
app.secret_key = "malaria-detection-secret-key"

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Render the upload page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload, run inference, and show result."""
    # Validate file presence
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a PNG, JPG, or BMP image.")
        return redirect(url_for("index"))

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        # Add timestamp to avoid overwriting
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{name}_{timestamp}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Run prediction
        result = predict_image(filepath)

        # Build the relative URL for the uploaded image
        image_url = url_for("static", filename=f"uploads/{unique_filename}")

        return render_template(
            "result.html",
            label=result["label"],
            confidence=result["confidence"],
            image_url=image_url,
            filename=unique_filename,
        )

    except Exception as exc:
        flash(f"Prediction failed: {exc}")
        return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🌐 Starting Malaria Detection Web App …")
    print("   Open http://127.0.0.1:5000 in your browser.\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
