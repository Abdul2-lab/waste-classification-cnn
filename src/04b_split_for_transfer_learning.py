"""
04b_split_for_transfer_learning.py
-------------------------------------
Transfer Learning ke liye ALAG data split banata hai — sirf REAL images
(koi augmentation nahi), taake dikha sakein ke Transfer Learning kam
data mein bhi kaisi performance deta hai.

Humare scratch CNN ne 4,200 augmented images use ki thin.
Ye script sirf ~1,041 real images use karti hai (70/15/15 split).
"""

import random
from pathlib import Path
from PIL import Image

random.seed(42)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SPLIT_DIR = Path(__file__).parent.parent / "data" / "split_transfer"
CLASSES = ["cardboard", "glass", "paper"]
IMG_SIZE = (96, 96)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def main():
    print("=" * 60)
    print("Creating train/val/test split for Transfer Learning")
    print("(Real images only, NO augmentation)")
    print("=" * 60)

    for cls in CLASSES:
        images = sorted((RAW_DIR / cls).glob("*.jpg"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            dst_dir = SPLIT_DIR / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                img = Image.open(f).convert("RGB").resize(IMG_SIZE)
                img.save(dst_dir / f.name, quality=90)

        print(f"{cls}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    print("\n✅ Done! Ab 09_transfer_learning.py chalayein.")


if __name__ == "__main__":
    main()
