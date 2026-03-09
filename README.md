# 🔬 Malaria Detection using Deep Learning

A complete deep-learning pipeline that classifies microscopic blood smear images as **Parasitized** or **Uninfected**, with a Flask web application for real-time predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Deep Learning Architecture](#deep-learning-architecture)
- [Installation](#installation)
- [Running the Flask Application](#running-the-flask-application)
- [Example Prediction](#example-prediction)
- [Evaluation Metrics](#evaluation-metrics)

---

## Overview

Malaria diagnosis traditionally requires microscopic examination of blood smear slides by trained pathologists — a process that is slow, labour-intensive, and error-prone, especially in rural and resource-limited areas.

This project automates malaria detection using deep learning. A convolutional neural network analyses cell images and outputs:
- **Prediction label** — *Parasitized* or *Uninfected*
- **Confidence score** — how confident the model is in its prediction

---

## Deep Learning Architecture

Two models are implemented:

### Model 1 — Custom CNN
| Layer | Details |
|-------|---------|
| Conv2D (×3 blocks) | 32 → 64 → 128 filters, 3×3, ReLU, same padding |
| MaxPooling2D | 2×2 after each Conv block |
| BatchNormalization | After each MaxPool |
| Dense | 256 → 128 neurons, ReLU |
| Dropout | 0.5 → 0.3 |
| Output | Dense(1, sigmoid) |

### Model 2 — MobileNetV2 Transfer Learning
| Layer | Details |
|-------|---------|
| Base | MobileNetV2 (ImageNet weights, frozen) |
| GlobalAveragePooling2D | Reduces spatial dimensions |
| Dense | 128 neurons, ReLU |
| Dropout | 0.5 |
| Output | Dense(1, sigmoid) |

**Training config:** Adam optimiser · Binary Crossentropy · EarlyStopping · ModelCheckpoint

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd malaria-detector-skillwallet

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Flask Application

Make sure you have a trained model at `models/malaria_model.h5` and you have placed the downloaded dataset correctly if you intend to evaluate against it.

```bash
python web_app/app.py
```

Then open **http://127.0.0.1:5001** in your browser.

1. Upload a microscopic blood smear image.
2. Click **Analyse Image**.
3. View the prediction and confidence score.

---

## Example Prediction

### CLI

```bash
python src/predict.py path/to/cell_image.png
```

### Output

```
🔬 Prediction : Parasitized
   Confidence : 96.47%
```

### Web Interface

| Upload | Result |
|--------|--------|
| Drag & drop a cell image | Displays: **Parasitized** or **Uninfected** with confidence % |

---

## Evaluation Metrics

After running `evaluate_model.py` the following are generated:

| Metric | Description |
|--------|------------|
| Accuracy | Overall correctness |
| Precision | True positives / predicted positives |
| Recall | True positives / actual positives |
| F1 Score | Harmonic mean of precision & recall |
| ROC AUC | Area under the ROC curve |

Plots saved in `outputs/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `training_curves.png`

---

## Technologies Used

- **TensorFlow / Keras** — Deep learning framework
- **OpenCV** — Image processing
- **scikit-learn** — Evaluation metrics
- **Matplotlib** — Visualisations
- **Flask** — Web application
- **Pillow** — Image handling

---

## License

This project is for educational purposes.
