"""
09_transfer_learning.py
-------------------------
Transfer Learning — MobileNetV2 (Feature Extraction) apne waste dataset pe.

Comparison ke liye: humara scratch-trained CNN 96.56% test accuracy deta hai,
4,200 training images (augmentation se) use karke.

Ye script sirf 1,041 REAL images (koi augmentation nahi) se train karta hai,
dekhte hain Transfer Learning kam data mein kya result deta hai.
"""

from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

SPLIT_DIR = Path(__file__).parent.parent / "data" / "split_transfer"
MODEL_DIR = Path(__file__).parent.parent / "model"
IMG_SIZE = (96, 96)
CLASSES = ["cardboard", "glass", "paper"]
BATCH_SIZE = 32


def get_generators():
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        SPLIT_DIR / "train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=True, seed=42,
    )
    val_gen = val_datagen.flow_from_directory(
        SPLIT_DIR / "val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=False,
    )
    return train_gen, val_gen


def build_transfer_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
    base_model.trainable = False  # Feature Extraction — freeze everything

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(3, activation="softmax"),
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("=" * 60)
    print("Loading data (1,041 real images, NO augmentation)")
    print("=" * 60)
    train_gen, val_gen = get_generators()
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    print("\n" + "=" * 60)
    print("Building Transfer Learning Model (MobileNetV2)")
    print("=" * 60)
    model = build_transfer_model()
    model.summary()

    trainable_params = sum([p.numpy().size for p in model.trainable_weights])
    print(f"\nTrainable parameters: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("Training (Feature Extraction)")
    print("=" * 60)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        train_gen, validation_data=val_gen, epochs=15,
        callbacks=[early_stop], verbose=2,
    )

    model.save(MODEL_DIR / "transfer_learning_model.keras")
    print(f"\nModel saved to: {MODEL_DIR / 'transfer_learning_model.keras'}")

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
