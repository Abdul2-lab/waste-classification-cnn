"""
app.py
-------
Flask Web App — Smart Waste Classification System

Features (assignment requirements):
- Image upload karo
- Waste type predict karo
- Prediction confidence dikhao
- Recyclable / Non-recyclable batao

NOTE: Ye version TensorFlow Lite (ai-edge-litert) use karta hai, poori
TensorFlow library nahi — kyunki Render free tier pe sirf 512MB RAM hai,
aur poori TensorFlow load hone mein 300-400MB kha jati hai jisse
"Out of Memory" crash hota tha. TFLite sirf ~40MB use karta hai.

Run: python3 app.py
Phir browser mein: http://localhost:5000
"""

import os
from pathlib import Path

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

try:
    # Production/lightweight import path
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    # Fallback agar sirf full tensorflow installed ho (jaise local dev mein)
    from tensorflow.lite.python.interpreter import Interpreter

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "waste_classifier.tflite"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (96, 96)
CLASSES = ["cardboard", "glass", "paper"]
RECYCLABLE_INFO = {"cardboard": True, "glass": True, "paper": True}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB max

print("Loading TFLite model...")
interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded successfully!")


def predict_image(image_path):
    # Image load + preprocess (resize, normalize) — bilkul training jaisa
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # TFLite interpreter ko input dena aur run karna
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]

    pred_idx = int(np.argmax(predictions))
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
