"""
app.py
-------
Flask Web App — Smart Waste Classification System

Features (assignment requirements):
- Image upload karo
- Waste type predict karo
- Prediction confidence dikhao
- Recyclable / Non-recyclable batao

Run: python3 app.py
Phir browser mein: http://localhost:5000
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "waste_classifier_best.keras"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (96, 96)
CLASSES = ["cardboard", "glass", "paper"]
RECYCLABLE_INFO = {"cardboard": True, "glass": True, "paper": True}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB max

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")


def predict_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    predictions = model.predict(arr, verbose=0)[0]
    pred_idx = np.argmax(predictions)
    pred_class = CLASSES[pred_idx]
    confidence = float(predictions[pred_idx])

    return {
        "class": pred_class,
        "confidence": round(confidence * 100, 2),
        "recyclable": RECYCLABLE_INFO[pred_class],
        "all_probabilities": {
            CLASSES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASSES))
        },
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = UPLOAD_DIR / filename
    file.save(filepath)

    try:
        result = predict_image(filepath)
        result["image_url"] = f"/static/uploads/{filename}"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
