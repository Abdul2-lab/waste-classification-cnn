"""
05_cnn_model.py
-----------------
CNN Architecture Definition

Architecture:
  Input (224x224x3)
    -> Conv2D(32) -> ReLU -> MaxPooling
    -> Conv2D(64) -> ReLU -> MaxPooling
    -> Conv2D(128) -> ReLU -> MaxPooling
    -> Flatten
    -> Dense(128) -> ReLU -> Dropout
    -> Dense(3) -> Softmax   (cardboard / glass / paper)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

IMG_SIZE = (96, 96)
NUM_CLASSES = 3


def build_cnn_model():
    model = Sequential(name="Waste_Classifier_CNN")

    # --- Convolution Block 1 ---
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="conv_block1"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_block1"))

    # --- Convolution Block 2 ---
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_block2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_block2"))

    # --- Convolution Block 3 ---
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_block3"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_block3"))

    # --- Convolution Block 4 ---
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_block4"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_block4"))

    # --- Fully Connected Layers ---
    model.add(Flatten(name="flatten"))
    model.add(Dense(128, activation="relu", name="fc1"))
    model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation="softmax", name="output"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
