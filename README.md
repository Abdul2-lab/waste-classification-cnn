# ♻️ Smart Waste Classification System — CNN Project

Ek complete, end-to-end **Convolutional Neural Network (CNN)** project jo waste images ko
teen categories mein classify karta hai: **Cardboard, Glass, Paper**.

Dataset: **TrashNet** (real-world waste images) — [garythung/trashnet](https://github.com/garythung/trashnet)

**Final Test Accuracy: 96.56%** 🎯

---

## 📊 Project Results Summary

| Metric | Value |
|---|---|
| Test Accuracy | 96.56% |
| Classes | Cardboard, Glass, Paper |
| Total Images (after augmentation) | 6,000 (2,000 per class) |
| Train / Val / Test Split | 4,200 / 900 / 900 (70/15/15) |
| Real source images | 1,488 (before augmentation) |

Per-class performance:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Cardboard | 0.98 | 0.95 | 0.96 |
| Glass | 0.96 | 0.98 | 0.97 |
| Paper | 0.96 | 0.97 | 0.96 |

---

## 📁 Project Structure

```
waste_classification_cnn/
├── data/
│   └── raw/                    ← Real TrashNet images (cardboard/glass/paper)
│       (data/augmented & data/split are NOT included — regenerate with scripts below)
├── src/
│   ├── 01_dataset_preparation.py   ← PHASE 1: corrupted/duplicate check, class distribution
│   ├── 02_eda.py                    ← PHASE 2: EDA (resolution, RGB, brightness, box plots)
│   ├── 03_preprocessing_augmentation.py  ← PHASE 3: resize, normalize, augment to 2000/class
│   ├── 04_train_val_test_split.py  ← PHASE 4: 70/15/15 split
│   ├── 05_cnn_model.py              ← CNN architecture definition
│   ├── 06_train.py                  ← Standard training script (single run)
│   ├── 06b_train_resumable.py       ← Resumable training (train in small increments)
│   ├── 07_evaluate.py               ← Test set evaluation + confusion matrix
│   ├── 08_predict.py                ← Command-line prediction on sample images
│   └── plot_history.py              ← Plot training curves from saved state
├── model/
│   ├── waste_classifier_best.keras  ← 🔑 Trained model (96.56% test accuracy)
│   └── training_state.json          ← Training history (13 epochs)
├── outputs/                         ← All EDA plots, training curves, confusion matrix
├── templates/index.html             ← Flask web UI
├── static/uploads/                  ← Uploaded images land here (runtime)
├── app.py                           ← Flask web app (upload image → get prediction)
└── requirements.txt
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start — Use the Already-Trained Model

Sabse aasan tareeqa: model already trained hai (`model/waste_classifier_best.keras`),
seedha web app chalayein:

```bash
python app.py
```

Browser mein kholein: **http://localhost:5000** — image upload karein aur prediction dekhein.

---

## 🔄 Full Pipeline Se Dobara Chalana (Reproduce From Scratch)

Agar aap poora pipeline khud dobara chalana chahte hain (data prep se training tak):

### Step 1: Real Data Download Karein
`data/raw/` mein already 1,488 real TrashNet images hain (cardboard, glass, paper).
Agar aapko fresh copy chahiye: [TrashNet GitHub](https://github.com/garythung/trashnet) se
`data/dataset-resized.zip` download kar ke `data/raw/<class>/` mein extract karein.

### Step 2: Phases Chalayein (order se)

```bash
cd src

python 01_dataset_preparation.py       # Corrupted/duplicate images check
python 02_eda.py                        # EDA graphs generate
python 03_preprocessing_augmentation.py # 2000 images/class tak augment (~5-10 min)
python 04_train_val_test_split.py       # 70/15/15 split
```

### Step 3: Model Train Karein

**Option A — Resumable (recommended, CPU pe safe):**
```bash
python 06b_train_resumable.py 3   # 3 epochs train karega, checkpoint save karega
python 06b_train_resumable.py 3   # phir se chalayein — agle 3 epochs se continue karega
# ... jab tak satisfied na ho jayein (humne 13 epochs mein 95.3% val accuracy paayi)
python plot_history.py            # training curves plot karega
```

**Option B — Ek hi baar mein (agar GPU hai ya time available hai):**
```bash
python 06_train.py
```

### Step 4: Evaluate Karein
```bash
python 07_evaluate.py    # Test set pe evaluate, confusion matrix + report generate
```

### Step 5: Predict Karein
```bash
python 08_predict.py     # Sample images pe command-line predictions
```

### Step 6: Web App Chalayein
```bash
cd ..
python app.py
```

---

## 🏗️ CNN Architecture

```
Input (96 x 96 x 3)
  → Conv2D(32) → BatchNorm → ReLU → MaxPooling
  → Conv2D(64) → BatchNorm → ReLU → MaxPooling
  → Conv2D(128) → BatchNorm → ReLU → MaxPooling
  → Conv2D(128) → BatchNorm → ReLU → MaxPooling
  → Flatten
  → Dense(128) → ReLU → Dropout(0.4)
  → Dense(3) → Softmax   (cardboard / glass / paper)
```

**Note:** Image size 96x96 use kiya gaya (assignment mein 224x224 suggested tha) taake
CPU-only environment mein training feasible ho. Agar GPU available ho, `src/05_cnn_model.py`
mein `IMG_SIZE = (224, 224)` kar sakte hain — architecture waisi hi rahegi.

- **Optimizer:** Adam (learning rate 0.0001 stabilization ke baad)
- **Loss:** Categorical Crossentropy
- **Total parameters:** ~730K

---

## 📈 Training Notes (Real Experience)

Training ke dauran epoch 6 pe instability aayi thi (val_accuracy 86% se gir kar 37% ho gayi,
val_loss spike hua) — ye batata hai ke **learning rate zyada** tha us waqt. Learning rate ko
0.0001 pe reduce karne ke baad training turant stabilize ho gayi aur consistently improve hoti
gayi (epoch 8 se 13 tak: 94.78% → 95.33% val accuracy). Ye ek real, common CNN training issue
hai — `outputs/08_training_curves.png` mein ye clearly dikhta hai.

Ye "resumable" training approach isliye use kiya gaya taake:
1. CPU-only training (jo slow hoti hai) ko chhote, manageable increments mein kiya ja sake
2. Best model automatically preserve ho, chahe baad ke epochs kharab ho jayein

---

## 🌐 Flask Web App Features

- Drag & drop ya click se image upload
- Real-time prediction (waste type + confidence %)
- Recyclable / Non-recyclable badge
- Sab 3 classes ki probability breakdown (progress bars)

---

## 📋 Assignment Phases Covered

| Phase | Status | Script |
|---|---|---|
| Phase 1 — Dataset Preparation | ✅ | `01_dataset_preparation.py` |
| Phase 2 — EDA | ✅ | `02_eda.py` |
| Phase 3 — Preprocessing + Augmentation | ✅ | `03_preprocessing_augmentation.py` |
| Phase 4 — Train/Val/Test Split | ✅ | `04_train_val_test_split.py` |
| CNN Model + Training | ✅ | `05_cnn_model.py`, `06b_train_resumable.py` |
| Evaluation | ✅ | `07_evaluate.py` |
| Deployment (Web App) | ✅ | `app.py` |

---

## 💡 Aage Kya Improve Ho Sakta Hai

1. **Transfer Learning** — pretrained model (MobileNetV2, ResNet50) use karke accuracy aur
   badhai ja sakti hai, khaas kar agar zyada classes add karni hon
2. **More classes** — TrashNet ke baaki 3 classes (metal, plastic, trash) bhi add kar sakte hain
3. **Larger image size** — GPU available ho to 224x224 pe train kar ke accuracy compare karein
4. **Model deployment** — Flask app ko Render/Railway pe host kar ke public URL bana sakte hain
