"""
07_evaluate.py
----------------
Best trained model ko TEST SET (jo model ne kabhi nahi dekha) pe evaluate karta hai.

Generates:
1. Test accuracy & loss
2. Confusion matrix (image)
3. Classification report (precision, recall, f1-score per class)
4. Misclassified examples grid (galat predictions dekhna)
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from importlib import import_module
cnn_module = import_module("05_cnn_model")
IMG_SIZE = cnn_module.IMG_SIZE

SPLIT_DIR = Path(__file__).parent.parent / "data" / "split"
MODEL_DIR = Path(__file__).parent.parent / "model"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
CLASSES = ["cardboard", "glass", "paper"]
BATCH_SIZE = 32


def main():
    print("=" * 60)
    print("Loading best model")
    print("=" * 60)
    model = load_model(MODEL_DIR / "waste_classifier_best.keras")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        SPLIT_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    print(f"Test samples: {test_gen.samples}")

    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss    : {test_loss:.4f}")

    # Predictions
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print("\nClassification Report:")
    print(report)

    # Save confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — Test Set (Accuracy: {test_accuracy:.2%})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_confusion_matrix.png", dpi=150)
    plt.close()
    print("\nSaved: outputs/09_confusion_matrix.png")

    # Save text report
    with open(OUTPUT_DIR / "evaluation_report.txt", "w") as f:
        f.write("WASTE CLASSIFICATION CNN — EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss    : {test_loss:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print("Saved: outputs/evaluation_report.txt")

    # Misclassified examples grid
    misclassified_idx = np.where(y_pred != y_true)[0]
    print(f"\nMisclassified: {len(misclassified_idx)} / {len(y_true)}")

    if len(misclassified_idx) > 0:
        n_show = min(8, len(misclassified_idx))
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes = axes.flatten()

        filepaths = [test_gen.filepaths[i] for i in misclassified_idx[:n_show]]
        for ax, idx, fp in zip(axes, misclassified_idx[:n_show], filepaths):
            from PIL import Image
            img = Image.open(fp)
            ax.imshow(img)
            ax.axis("off")
            true_label = CLASSES[y_true[idx]]
            pred_label = CLASSES[y_pred[idx]]
            confidence = y_pred_proba[idx][y_pred[idx]]
            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.0%})", fontsize=9, color="red")

        for ax in axes[n_show:]:
            ax.axis("off")

        plt.suptitle("Misclassified Examples", fontsize=13)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "10_misclassified_examples.png", dpi=150)
        plt.close()
        print("Saved: outputs/10_misclassified_examples.png")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
