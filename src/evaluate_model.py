"""
evaluate_model.py
==================
Evaluates the trained malaria detection model and generates:

* Classification metrics — Accuracy, Precision, Recall, F1 Score, ROC AUC
* Visualisations        — Confusion Matrix, Training curves, ROC Curve

Usage
-----
    python src/evaluate_model.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for servers / CI)
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from tensorflow.keras.models import load_model

from preprocess_data import get_data_generators

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "malaria_model.h5")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_model(model_path: str = MODEL_PATH):
    """
    Load the trained model, run predictions on the validation set, and
    print / save evaluation metrics + charts.
    """
    _ensure_output_dir()

    # 1. Load model & data
    print(f"📂 Loading model from {model_path} …")
    model = load_model(model_path)

    _, val_gen = get_data_generators()
    val_gen.reset()

    # 2. Predictions
    print("🔍 Running predictions on validation set …")
    y_pred_prob = model.predict(val_gen, verbose=1).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = val_gen.classes

    # 3. Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_prob)

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }

    print("\n" + "=" * 50)
    print("          MODEL EVALUATION RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:>12s} : {v}")
    print("=" * 50)
    print("\nClassification Report:\n")
    class_names = list(val_gen.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save metrics JSON
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to {metrics_path}")

    # 4. Visualisations
    _plot_confusion_matrix(y_true, y_pred, class_names)
    _plot_roc_curve(y_true, y_pred_prob)

    print(f"\n📊 All plots saved under {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def _plot_confusion_matrix(y_true, y_pred, class_names):
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   ✅  Confusion matrix → {path}")


def _plot_roc_curve(y_true, y_pred_prob):
    """Save an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_val = roc_auc_score(y_true, y_pred_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#0077b6", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   ✅  ROC curve       → {path}")


def plot_training_history(history):
    """
    Plot training & validation accuracy and loss curves.

    Call this after ``train()`` returns a ``History`` object.
    You can also call it standalone by loading a saved history JSON.
    """
    _ensure_output_dir()

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs_range, acc, "o-", label="Train Accuracy", color="#0077b6")
    ax1.plot(epochs_range, val_acc, "o-", label="Val Accuracy", color="#e63946")
    ax1.set_title("Accuracy vs Epochs", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs_range, loss, "o-", label="Train Loss", color="#0077b6")
    ax2.plot(epochs_range, val_loss, "o-", label="Val Loss", color="#e63946")
    ax2.set_title("Loss vs Epochs", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   ✅  Training curves → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate_model()
