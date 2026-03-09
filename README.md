# 🔬 Malaria Detection using Deep Learning

A complete deep-learning pipeline that classifies microscopic blood smear images as **Parasitized** or **Uninfected**, with a Flask web application for real-time predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Deep Learning Architecture](#deep-learning-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training on Google Colab](#training-on-google-colab)
- [Training Locally](#training-locally)
- [Running the Flask Application](#running-the-flask-application)
- [Example Prediction](#example-prediction)

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

## Dataset

**NIH Malaria Cell Images Dataset** — ~27,558 microscopic cell images.

| Class | Description | Count |
|-------|------------|-------|
| Parasitized | Malaria-positive cells | ~13,779 |
| Uninfected | Healthy cells | ~13,779 |

The dataset is **automatically downloaded** by the project scripts.

Source: [NIH / NLM](https://ceb.nlm.nih.gov/repositories/malaria-datasets/)

---

## Project Structure

```
malaria-detector-skillwallet/
├── dataset/                    # Auto-downloaded images
│   ├── Parasitized/
│   └── Uninfected/
├── models/
│   └── malaria_model.h5        # Trained model (generated)
├── notebooks/
│   └── train_model_colab.ipynb  # Google Colab notebook
├── src/
│   ├── download_dataset.py     # Dataset downloader
│   ├── preprocess_data.py      # Image preprocessing & augmentation
│   ├── train_model.py          # Model training (CNN / MobileNetV2)
│   ├── evaluate_model.py       # Metrics & visualisations
│   └── predict.py              # Single-image inference
├── web_app/
│   ├── app.py                  # Flask application
│   └── templates/
│       ├── index.html          # Upload page
│       └── result.html         # Prediction result page
├── static/
│   └── uploads/                # Uploaded images
├── outputs/                    # Evaluation plots (generated)
├── requirements.txt
└── README.md
```

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

## Training on Google Colab

1. Open **`notebooks/train_model_colab.ipynb`** in [Google Colab](https://colab.research.google.com/).
2. Go to **Runtime → Change runtime type → GPU (T4)**.
3. Run all cells — the notebook will:
   - Download the dataset automatically
   - Preprocess & augment images
   - Build and train the model
   - Evaluate and generate plots
   - Save `malaria_model.h5`
4. Download the trained model and place it in `models/malaria_model.h5`.

---

## Training Locally

```bash
# 1. Download the dataset
python src/download_dataset.py

# 2. Train with MobileNetV2 (default)
cd src
python train_model.py --model mobilenet --epochs 10

# Or train with the custom CNN
python train_model.py --model cnn --epochs 15

# 3. Evaluate the model
python evaluate_model.py
```

> ⚠️ Training on CPU (e.g. Mac without GPU) will be slow. Google Colab with GPU is recommended.

---

## Running the Flask Application

Make sure you have a trained model at `models/malaria_model.h5`.

```bash
python web_app/app.py
```

Then open **http://127.0.0.1:5001** in your browser.

1. Upload a microscopic blood smear image.
2. Click **Analyse Image**.
3. View the prediction and confidence score.

---

## Deployment (Render)

Machine Learning applications rely on large dependencies like TensorFlow, making them unsuitable for Serverless platforms like Vercel (which has a strict 250MB limit). We recommend a container-based app deployment service like **Render**.

**Steps to deploy to Render:**

1. Commit your code to a GitHub repository.
2. Sign up on [Render](https://render.com/) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Configure the service:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn web_app.app:app`
5. Click **Create Web Service**. Render will build and deploy your Malaria Detection app.

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

This project is for educational purposes. The NIH Malaria Dataset is in the public domain.
