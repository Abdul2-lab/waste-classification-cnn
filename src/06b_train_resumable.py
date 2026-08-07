"""
06b_train_resumable.py
------------------------
Resumable CNN training — trains a small number of epochs per run,
saves a checkpoint + history, and can be re-invoked to continue.

Usage: python3 06b_train_resumable.py <epochs_this_run>
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from importlib import import_module
cnn_module = import_module("05_cnn_model")
build_cnn_model = cnn_module.build_cnn_model
IMG_SIZE = cnn_module.IMG_SIZE

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras import backend as K

SPLIT_DIR = Path(__file__).parent.parent / "data" / "split"
MODEL_DIR = Path(__file__).parent.parent / "model"
STATE_FILE = MODEL_DIR / "training_state.json"
CHECKPOINT_PATH = MODEL_DIR / "waste_classifier_checkpoint.keras"
BEST_MODEL_PATH = MODEL_DIR / "waste_classifier_best.keras"
BATCH_SIZE = 32


class SaveBestAcrossRuns(Callback):
    """Har epoch ke baad check karta hai — agar val_accuracy ne ab tak ka
    best beat kar diya hai (state.json ke across-run record ke against),
    to turant model ko best_model_path pe save kar deta hai.
    Isse agar baad ke epochs kharab ho jayein, best weights safe rehte hain."""

    def __init__(self, state, save_path):
        super().__init__()
        self.state = state
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy", 0)
        if val_acc > self.state["best_val_accuracy"]:
            self.state["best_val_accuracy"] = val_acc
            self.model.save(self.save_path)
            print(f"  -> New best val_accuracy: {val_acc:.4f}. Saved to {self.save_path.name}")


def get_generators():
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        SPLIT_DIR / "train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=["cardboard", "glass", "paper"],
        shuffle=True, seed=42,
    )
    val_gen = val_datagen.flow_from_directory(
        SPLIT_DIR / "val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=["cardboard", "glass", "paper"],
        shuffle=False,
    )
    return train_gen, val_gen


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"epochs_done": 0, "history": {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []},
            "best_val_accuracy": 0.0}


def save_state(state):
    MODEL_DIR.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    epochs_this_run = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    state = load_state()
    print(f"Resuming from epoch {state['epochs_done']}. Training {epochs_this_run} more epochs.")

    train_gen, val_gen = get_generators()

    if CHECKPOINT_PATH.exists():
        print("Loading existing checkpoint...")
        model = load_model(CHECKPOINT_PATH)
        # Learning rate kam kar dete hain — stability ke liye (recompile with fresh optimizer)
        from tensorflow.keras.optimizers import Adam
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
        print("Learning rate set to: 0.0001")
    else:
        print("No checkpoint found, building new model...")
        model = build_cnn_model()

    callbacks = [
        SaveBestAcrossRuns(state, BEST_MODEL_PATH),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_this_run,
        callbacks=callbacks,
        verbose=2,
    )

    # Update state
    for key in ["loss", "accuracy", "val_loss", "val_accuracy"]:
        state["history"][key].extend(history.history[key])
    state["epochs_done"] += epochs_this_run

    save_state(state)
    model.save(CHECKPOINT_PATH)

    print(f"\nCheckpoint saved. Total epochs done: {state['epochs_done']}")
    print(f"Latest val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Best val_accuracy so far (all runs): {state['best_val_accuracy']:.4f}")
    print(f"Latest val_loss: {history.history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
