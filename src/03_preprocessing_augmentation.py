"""
03_preprocessing_augmentation.py
----------------------------------
PHASE 3 — Image Preprocessing + Data Augmentation

Ye script:
1. Har image ko resize karta hai (224 x 224)
2. Data Augmentation apply karta hai (rotation, flip, zoom, shift, brightness, shear)
3. Har class ko TARGET_PER_CLASS (2000) images tak expand karta hai
   (real images + augmented copies)

NOTE: Normalize (pixel / 255) training ke waqt hoga (ImageDataGenerator mein),
      taake disk pe images normal 0-255 JPG format mein hi save rahein.
"""

from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
AUG_DIR = Path(__file__).parent.parent / "data" / "augmented"
CLASSES = ["cardboard", "glass", "paper"]
IMG_SIZE = (224, 224)
TARGET_PER_CLASS = 2000  # Har class ko itni images tak expand karna hai

# Augmentation configuration - assignment mein suggested sab techniques
datagen = ImageDataGenerator(
    rotation_range=30,          # Rotation
    width_shift_range=0.15,     # Width shift
    height_shift_range=0.15,    # Height shift
    zoom_range=0.2,             # Zoom
    horizontal_flip=True,       # Horizontal flip
    brightness_range=[0.7, 1.3],# Brightness adjustment
    shear_range=0.15,           # Shear
    fill_mode="nearest",
)


def resize_and_copy_real_images(cls):
    """Real images ko 224x224 resize kar ke augmented folder mein copy karta hai."""
    src_dir = RAW_DIR / cls
    dst_dir = AUG_DIR / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(src_dir.glob("*.jpg")):
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        img.save(dst_dir / f"real_{count:04d}.jpg", quality=90)
        count += 1

    return count


def augment_to_target(cls, real_count):
    """Data augmentation se images generate karta hai jab tak TARGET_PER_CLASS na pahunch jaye."""
    src_dir = RAW_DIR / cls
    dst_dir = AUG_DIR / cls

    needed = TARGET_PER_CLASS - real_count
    if needed <= 0:
        print(f"  {cls}: already has {real_count} >= {TARGET_PER_CLASS}, no augmentation needed")
        return 0

    real_images = sorted(src_dir.glob("*.jpg"))
    generated = 0
    img_idx = 0

    while generated < needed:
        img_path = real_images[img_idx % len(real_images)]
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        arr = img_to_array(img)
        arr = arr.reshape((1,) + arr.shape)

        # Ek augmented version generate karo
        for batch in datagen.flow(arr, batch_size=1):
            aug_img = array_to_img(batch[0])
            aug_img.save(dst_dir / f"aug_{generated:05d}.jpg", quality=90)
            generated += 1
            break  # Sirf 1 augmented image per iteration

        img_idx += 1

    return generated


def main():
    print("=" * 60)
    print("PHASE 3: Image Preprocessing + Data Augmentation")
    print(f"Target per class: {TARGET_PER_CLASS} images")
    print("=" * 60)

    AUG_DIR.mkdir(exist_ok=True)
    summary = {}

    for cls in CLASSES:
        print(f"\nProcessing class: {cls}")
        real_count = resize_and_copy_real_images(cls)
        print(f"  Resized & copied {real_count} real images (224x224)")

        aug_count = augment_to_target(cls, real_count)
        print(f"  Generated {aug_count} augmented images")

        total = real_count + aug_count
        summary[cls] = {"real": real_count, "augmented": aug_count, "total": total}
        print(f"  Total for '{cls}': {total} images")

    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    for cls, stats in summary.items():
        print(f"  {cls:12s} -> real: {stats['real']:4d}  augmented: {stats['augmented']:4d}  total: {stats['total']:4d}")

    print(f"\nGrand total images: {sum(s['total'] for s in summary.values())}")
    print("\n✅ Phase 3 complete. Ab 04_train_val_test_split.py chalayein.")


if __name__ == "__main__":
    main()
