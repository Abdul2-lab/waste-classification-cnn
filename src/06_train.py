"""
06_train.py
------------
CNN Model Training

Ye script:
1. Train/Val data ko ImageDataGenerator se load karta hai (normalize: pixel/255)
2. CNN model banata hai (05_cnn_model.py se)
3. Model train karta hai (EarlyStopping ke saath)
4. Training curves (accuracy/loss) save karta hai
5. Trained model disk pe save karta hai
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from importlib import import_module
cnn_module = import_module("05_cnn_model")
build_cnn_model = cnn_module.build_cnn_model
IMG_SIZE = cnn_module.IMG_SIZE

SPLIT_DIR = Path(__file__).parent.parent / "data" / "split"
MODEL_DIR = Path(__file__).parent.parent / "model"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
BATCH_SIZE = 32
EPOCHS = 15


def get_generators():
    # Training data - normalize hi karte hain (augmentation already disk pe ho chuki hai)
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        SPLIT_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=["cardboard", "glass", "paper"],
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        SPLIT_DIR / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=["cardboard", "glass", "paper"],
        shuffle=False,
    )

    return train_gen, val_gen


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["loss"], label="Training Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Training Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Loading data generators")
    print("=" * 60)
    train_gen, val_gen = get_generators()
    print(f"Class indices: {train_gen.class_indices}")
    print(f"Training samples  : {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    print("\n" + "=" * 60)
    print("Building CNN model")
    print("=" * 60)
    model = build_cnn_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(
            str(MODEL_DIR / "waste_classifier_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print("\n" + "=" * 60)
    print("Training model")
    print("=" * 60)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    # Final model bhi save karo
    model.save(MODEL_DIR / "waste_classifier_final.keras")
    print(f"\nModel saved to: {MODEL_DIR / 'waste_classifier_final.keras'}")
    print(f"Best model saved to: {MODEL_DIR / 'waste_classifier_best.keras'}")

    plot_training_curves(history, OUTPUT_DIR / "08_training_curves.png")

    print("\n✅ Training complete. Ab 07_evaluate.py chalayein.")


if __name__ == "__main__":
    main()
