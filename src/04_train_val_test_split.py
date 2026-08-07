"""
04_train_val_test_split.py
----------------------------
PHASE 4 — Train / Validation / Test Split

Suggested split (from assignment):
  Training   70%
  Validation 15%
  Testing    15%

Har class ki 2000 images ko is split ke hisaab se
data/split/train, data/split/val, data/split/test mein organize karta hai.
"""

import random
import shutil
from pathlib import Path

AUG_DIR = Path(__file__).parent.parent / "data" / "augmented"
SPLIT_DIR = Path(__file__).parent.parent / "data" / "split"
CLASSES = ["cardboard", "glass", "paper"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # Reproducibility ke liye


def split_class(cls):
    src_dir = AUG_DIR / cls
    all_images = sorted(src_dir.glob("*.jpg"))
    random.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # Test ko remaining images mil jayengi (rounding errors avoid karne ke liye)

    train_files = all_images[:n_train]
    val_files = all_images[n_train:n_train + n_val]
    test_files = all_images[n_train + n_val:]

    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        dst_dir = SPLIT_DIR / split_name / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, dst_dir / f.name)

    return len(train_files), len(val_files), len(test_files)


def main():
    print("=" * 60)
    print("PHASE 4: Train / Validation / Test Split (70/15/15)")
    print("=" * 60)

    summary = {}
    for cls in CLASSES:
        n_train, n_val, n_test = split_class(cls)
        summary[cls] = (n_train, n_val, n_test)
        print(f"  {cls:12s} -> train: {n_train:4d}  val: {n_val:4d}  test: {n_test:4d}")

    total_train = sum(s[0] for s in summary.values())
    total_val = sum(s[1] for s in summary.values())
    total_test = sum(s[2] for s in summary.values())

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Total Train: {total_train}")
    print(f"  Total Val  : {total_val}")
    print(f"  Total Test : {total_test}")
    print(f"  Grand Total: {total_train + total_val + total_test}")

    print("\n✅ Phase 4 complete. Ab CNN model training shuru ho sakti hai (05_cnn_model.py + 06_train.py).")


if __name__ == "__main__":
    main()
