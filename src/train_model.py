"""
train_model.py
===============
Trains deep-learning models for malaria cell classification.

Two architectures are available:
  1. **Custom CNN** — lightweight convolutional network built from scratch.
  2. **MobileNetV2** — transfer-learning model with frozen ImageNet weights
     and a custom classification head.

Usage
-----
    # Train the default model (MobileNetV2)
    python src/train_model.py

    # Train the custom CNN
    python src/train_model.py --model cnn

    # Adjust epochs / batch size
    python src/train_model.py --model mobilenet --epochs 15 --batch_size 32
"""

import os
import argparse

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

# Local imports
from preprocess_data import get_data_generators

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "malaria_model.h5")

IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_custom_cnn(input_shape=(224, 224, 3)) -> models.Model:
    """
    Build a custom CNN for binary classification.

    Architecture:
        Conv2D → ReLU → MaxPool → BatchNorm  (×3 blocks)
        Flatten → Dense → Dropout → Dense(1, sigmoid)
    """
    model = models.Sequential(name="Custom_CNN")

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu",
                            padding="same", input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())

    # Classification head
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


def build_mobilenet_v2(input_shape=(224, 224, 3)) -> models.Model:
    """
    Build a MobileNetV2-based transfer-learning model.

    Steps:
        1. Load MobileNetV2 without top (pretrained on ImageNet).
        2. Freeze the base layers.
        3. Add a custom classification head:
           GlobalAveragePooling → Dense(128) → Dropout → Dense(1, sigmoid)
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    # Freeze base layers
    base_model.trainable = False

    model = models.Sequential(name="MobileNetV2_Transfer")
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------
def train(model_type: str = "mobilenet", epochs: int = 10, batch_size: int = 32):
    """
    Compile and train the selected model.

    Parameters
    ----------
    model_type : str
        ``"cnn"`` for Custom CNN, ``"mobilenet"`` for MobileNetV2.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Batch size for data generators.
    """
    # Data
    train_gen, val_gen = get_data_generators(batch_size=batch_size)

    # Model
    if model_type == "cnn":
        print("\n🧠 Building Custom CNN …")
        model = build_custom_cnn()
    else:
        print("\n🧠 Building MobileNetV2 Transfer-Learning model …")
        model = build_mobilenet_v2()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train
    print(f"\n🚀 Training for up to {epochs} epochs …\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=cb_list,
    )

    print(f"\n✅ Best model saved to {MODEL_PATH}")
    return history, model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a malaria detection model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet",
        choices=["cnn", "mobilenet"],
        help="Model architecture: 'cnn' or 'mobilenet' (default: mobilenet)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check for GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"🖥  GPU detected: {gpus}")
    else:
        print("🖥  No GPU detected — training on CPU (this may be slow).")

    train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
