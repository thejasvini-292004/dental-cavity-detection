"""
Local desktop inference (no web UI).
====================================
Opens a file dialog, lets you pick a dental X-ray, runs the U-Net model and
shows the original image, probability map, binary mask and a detection overlay
in a matplotlib window.

Usage:  python src/predict_local.py
"""

import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Model lives at the repository root, next to this script's parent folder.
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "unet_cavity_final.h5"
IMG_SIZE = 128


def main():
    print("Loading model…")
    # compile=False -> inference only, no need for the training loss/metrics.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded.")

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Dental Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")],
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    orig = cv2.imread(file_path)
    if orig is None:
        print("Error: could not read image.")
        return

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)) / 255.0
    prob = model.predict(small[np.newaxis, ..., np.newaxis])[0, ..., 0]

    binary = (prob > 0.5).astype(np.uint8)
    binary_full = cv2.resize(binary, (orig.shape[1], orig.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    result = orig.copy()
    detected = False
    contours, _ = cv2.findContours(binary_full, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 100:
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result, "Cavity Detected", (x, max(y - 20, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            detected = True

    cv2.putText(result, f"File: {os.path.basename(file_path)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    plt.figure(figsize=(16, 6))
    for i, (title, img, cmap) in enumerate([
        ("Original", cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), None),
        ("Predicted probability", prob, "magma"),
        ("Binary mask", binary_full, "gray"),
        ("Detection", cv2.cvtColor(result, cv2.COLOR_BGR2RGB), None),
    ], start=1):
        plt.subplot(1, 4, i)
        plt.title(title)
        plt.imshow(img, cmap=cmap)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("\nRESULT:", "CAVITY DETECTED" if detected else "NO CAVITY DETECTED")


if __name__ == "__main__":
    main()
