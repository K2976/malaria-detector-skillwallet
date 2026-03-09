import os
import sys

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

from tensorflow.keras.models import load_model

try:
    model = load_model("models/malaria_detector.h5")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Success")
except Exception as e:
    print("Error:", e)
