"""
plot_history.py
-----------------
training_state.json se saved history load kar ke training curves plot karta hai.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = Path(__file__).parent.parent / "model"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
STATE_FILE = MODEL_DIR / "training_state.json"


def main():
    with open(STATE_FILE) as f:
        state = json.load(f)

    history = state["history"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["loss"], label="Training Loss", marker="o", markersize=3)
    axes[0].plot(history["val_loss"], label="Validation Loss", marker="o", markersize=3)
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["accuracy"], label="Training Accuracy", marker="o", markersize=3)
    axes[1].plot(history["val_accuracy"], label="Validation Accuracy", marker="o", markersize=3)
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_training_curves.png", dpi=150)
    plt.close()
    print(f"Saved: outputs/08_training_curves.png")
    print(f"Total epochs trained: {state['epochs_done']}")
    print(f"Best validation accuracy: {state['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
