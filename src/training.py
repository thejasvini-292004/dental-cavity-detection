# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


TRAIN_IMG_DIR = "/content/drive/MyDrive/teeth/patch_dataset/train/images"
TRAIN_MSK_DIR = "/content/drive/MyDrive/teeth/patch_dataset/train/masks"

VAL_IMG_DIR   = "/content/drive/MyDrive/teeth/patch_dataset/val/images"
VAL_MSK_DIR   = "/content/drive/MyDrive/teeth/patch_dataset/val/masks"

SAVE_BEST  = "/content/drive/MyDrive/teeth/unet_best_model.h5"
SAVE_FINAL = "/content/drive/MyDrive/teeth/unet_cavity_final.h5"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4


def load_data(img_dir, mask_dir):
    images = []
    masks = []

    files = sorted(os.listdir(img_dir))

    for fname in files:
        img_path = os.path.join(img_dir, fname)
        msk_path = os.path.join(mask_dir, fname)

        if not os.path.exists(msk_path):
            continue

        img = cv2.imread(img_path, 0)
        msk = cv2.imread(msk_path, 0)

        if img is None or msk is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE)) / 255.0
        msk = (msk > 0.5).astype(np.float32)

        img = img[..., np.newaxis]
        msk = msk[..., np.newaxis]

        images.append(img)
        masks.append(msk)

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

print("Loading training data...")
X_train, y_train = load_data(TRAIN_IMG_DIR, TRAIN_MSK_DIR)

print("Loading validation data...")
X_val, y_val = load_data(VAL_IMG_DIR, VAL_MSK_DIR)

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)


def augment_data(images, masks):
    for i in range(len(images)):
        if np.random.rand() < 0.5:
            images[i] = np.fliplr(images[i])
            masks[i]  = np.fliplr(masks[i])

        if np.random.rand() < 0.5:
            images[i] = np.flipud(images[i])
            masks[i]  = np.flipud(masks[i])

    return images, masks


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = layers.Input(input_shape)

 
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D()(c3)


    c4 = layers.Conv2D(256, 3, activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(256, 3, activation="relu", padding="same")(c4)


    u5 = layers.UpSampling2D()(c4)
    u5 = layers.Concatenate()([u5, c3])
    c5 = layers.Conv2D(128, 3, activation="relu", padding="same")(u5)
    c5 = layers.Conv2D(128, 3, activation="relu", padding="same")(c5)

    u6 = layers.UpSampling2D()(c5)
    u6 = layers.Concatenate()([u6, c2])
    c6 = layers.Conv2D(64, 3, activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(64, 3, activation="relu", padding="same")(c6)

    u7 = layers.UpSampling2D()(c6)
    u7 = layers.Concatenate()([u7, c1])
    c7 = layers.Conv2D(32, 3, activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(32, 3, activation="relu", padding="same")(c7)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c7)

    return models.Model(inputs, outputs)


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return loss

def focal_dice_loss(y_true, y_pred):
    return focal_loss()(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))


model = build_unet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=focal_dice_loss,
    metrics=[
        dice_coef,
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    SAVE_BEST,
    monitor="dice_coef",  
    save_best_only=True,
    mode="max",
    verbose=0
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="dice_coef",
    patience=15,
    mode="max",
    restore_best_weights=True,
    verbose=0
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="dice_coef",
    factor=0.5,
    patience=7,
    mode="max",
    verbose=0
)


for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    X_train_aug, y_train_aug = augment_data(X_train.copy(), y_train.copy())

    model.fit(
        X_train_aug, y_train_aug,
        epochs=1,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1  
    )


model.save(SAVE_FINAL)

print("\nTraining completed successfully.")
print("Best model saved at :", SAVE_BEST)
print("Final model saved at:", SAVE_FINAL)


print("\nEvaluating on validation set (final result only):")
results = model.evaluate(X_val, y_val, verbose=0)

print(f"Final Dice     : {results[1]:.4f}")
print(f"Final Precision: {results[2]:.4f}")
print(f"Final Recall   : {results[3]:.4f}")


files = sorted(os.listdir(VAL_IMG_DIR))[:5]

for f in files:
    img = cv2.imread(os.path.join(VAL_IMG_DIR, f), 0)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input = img_resized[np.newaxis, ..., np.newaxis]

    pred = model.predict(img_input)[0, ..., 0]
    pred = (pred > 0.5).astype(np.uint8)

    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Predicted Mask")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Ground Truth")
    gt = cv2.imread(os.path.join(VAL_MSK_DIR, f), 0)
    plt.imshow(gt, cmap="gray")
    plt.axis("off")

    plt.show()
