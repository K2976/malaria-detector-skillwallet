"""
preprocess_data.py
===================
Preprocesses the malaria cell-images dataset for training.

Steps performed
---------------
1. Resize all images to 224 × 224 pixels.
2. Normalise pixel values to [0, 1].
3. Apply data augmentation (rotation, horizontal flip, zoom, width/height shift).
4. Split the data 80 / 20 into training and validation generators.

Uses ``tf.keras.preprocessing.image.ImageDataGenerator``.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_data_generators(
    dataset_dir: str = DATASET_DIR,
    image_size: tuple = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
):
    """
    Create training and validation ``DirectoryIterator`` objects.

    Parameters
    ----------
    dataset_dir : str
        Path to the root dataset directory containing ``Parasitized/`` and
        ``Uninfected/`` sub-folders.
    image_size : tuple
        Target (height, width) for resizing.
    batch_size : int
        Number of images per batch.
    validation_split : float
        Fraction of data reserved for validation (0–1).

    Returns
    -------
    train_generator, val_generator : DirectoryIterator
    """

    # Training generator — with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    # Validation generator — only rescaling (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=validation_split,
    )

    print(f"📂 Loading images from: {dataset_dir}")
    print(f"   Image size : {image_size}")
    print(f"   Batch size : {batch_size}")
    print(f"   Val split  : {validation_split}")

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_generator = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    print(f"\n   Training samples   : {train_generator.samples}")
    print(f"   Validation samples : {val_generator.samples}")
    print(f"   Classes            : {train_generator.class_indices}\n")

    return train_generator, val_generator


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_gen, val_gen = get_data_generators()
    # Peek at one batch
    images, labels = next(train_gen)
    print(f"Batch shape : {images.shape}")
    print(f"Label sample: {labels[:8]}")
