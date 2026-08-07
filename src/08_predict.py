"""
08_predict.py
---------------
Trained model use karke KISI BHI naye image pe prediction deta hai.
Bataega: waste type, confidence %, aur recyclable hai ya nahi.
"""

import sys
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
cnn_module = import_module("05_cnn_model")
IMG_SIZE = cnn_module.IMG_SIZE

MODEL_DIR = Path(__file__).parent.parent / "model"
CLASSES = ["cardboard", "glass", "paper"]

# Recyclability info — assignment requirement
RECYCLABLE_INFO = {
    "cardboard": True,
    "glass": True,
    "paper": True,
}


def load_waste_model():
    return load_model(MODEL_DIR / "waste_classifier_best.keras")


def predict_image(model, image_path):
    """Ek image ka path le kar prediction, confidence, aur recyclability deta hai."""
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    predictions = model.predict(arr, verbose=0)[0]
    pred_idx = np.argmax(predictions)
    pred_class = CLASSES[pred_idx]
    confidence = predictions[pred_idx]

    is_recyclable = RECYCLABLE_INFO[pred_class]

    return {
        "class": pred_class,
        "confidence": float(confidence),
        "recyclable": is_recyclable,
        "all_probabilities": {CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))},
    }


def main():
    print("=" * 60)
    print("Loading trained model...")
    print("=" * 60)
    model = load_waste_model()
    print("Model loaded!\n")

    # Demo: test set se kuch sample images pe predict karte hain
    SPLIT_DIR = Path(__file__).parent.parent / "data" / "split" / "test"

    print("=" * 60)
    print("Sample Predictions")
    print("=" * 60)

    for cls in CLASSES:
        sample_dir = SPLIT_DIR / cls
        sample_file = sorted(sample_dir.glob("*.jpg"))[0]

        result = predict_image(model, sample_file)

        print(f"\nImage: {sample_file.name} (actual class: {cls})")
        print(f"  Predicted    : {result['class']}")
        print(f"  Confidence   : {result['confidence']*100:.2f}%")
        print(f"  Recyclable   : {'Yes' if result['recyclable'] else 'No'}")
        print(f"  All probabilities:")
        for c, p in result["all_probabilities"].items():
            print(f"    {c:12s}: {p*100:.2f}%")


if __name__ == "__main__":
    main()
