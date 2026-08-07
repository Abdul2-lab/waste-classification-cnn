"""
01_dataset_preparation.py
--------------------------
PHASE 1 — Dataset Preparation

Ye script:
1. Har class folder mein images check karta hai (corrupted images dhoondta hai)
2. Duplicate images detect karta hai (perceptual hashing se)
3. Image labels verify karta hai (folder name = class label)
4. Class distribution analyze + graph banata hai
5. Sample images har class se dikhata hai
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import imagehash

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
CLASSES = ["cardboard", "glass", "paper"]


def check_corrupted_images():
    """Har image ko open kar ke verify karta hai ke wo corrupt to nahi."""
    print("=" * 60)
    print("STEP 1: Checking for corrupted images")
    print("=" * 60)
    corrupted = []
    total_checked = 0

    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        for img_path in cls_dir.glob("*.jpg"):
            total_checked += 1
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Corruption check karta hai bina fully load kiye
            except Exception as e:
                corrupted.append(img_path)
                print(f"  Corrupted: {img_path.name} ({e})")

    print(f"\nTotal checked: {total_checked}")
    print(f"Corrupted found: {len(corrupted)}")

    # Corrupted images remove kar dete hain agar mile
    for path in corrupted:
        path.unlink()
        print(f"  Removed: {path.name}")

    return len(corrupted)


def check_duplicate_images():
    """Perceptual hashing use kar ke duplicate/near-duplicate images dhoondta hai."""
    print("\n" + "=" * 60)
    print("STEP 2: Checking for duplicate images")
    print("=" * 60)

    total_duplicates = 0

    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        hashes = {}
        duplicates_in_class = []

        for img_path in sorted(cls_dir.glob("*.jpg")):
            with Image.open(img_path) as img:
                h = str(imagehash.average_hash(img))
            if h in hashes:
                duplicates_in_class.append(img_path)
            else:
                hashes[h] = img_path

        print(f"  {cls}: {len(duplicates_in_class)} duplicates found")
        total_duplicates += len(duplicates_in_class)

        # Duplicates remove karo
        for path in duplicates_in_class:
            path.unlink()

    print(f"\nTotal duplicates removed: {total_duplicates}")
    return total_duplicates


def verify_labels():
    """Confirm karta hai ke har class folder mein sirf valid image files hain."""
    print("\n" + "=" * 60)
    print("STEP 3: Verifying labels (folder structure)")
    print("=" * 60)

    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        count = len(list(cls_dir.glob("*.jpg")))
        print(f"  Class '{cls}': {count} valid images -> label = '{cls}'")


def analyze_class_distribution():
    """Class distribution ka bar chart aur pie chart banata hai."""
    print("\n" + "=" * 60)
    print("STEP 4: Class Distribution Analysis")
    print("=" * 60)

    counts = {}
    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        counts[cls] = len(list(cls_dir.glob("*.jpg")))
        print(f"  {cls}: {counts[cls]} images")

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    axes[0].bar(counts.keys(), counts.values(), color=["#C9A24B", "#5DBEA3", "#7C8FA6"])
    axes[0].set_title("Images per Class (Bar Chart)")
    axes[0].set_ylabel("Number of Images")
    for i, (k, v) in enumerate(counts.items()):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%",
                colors=["#C9A24B", "#5DBEA3", "#7C8FA6"])
    axes[1].set_title("Class Distribution (Pie Chart)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_class_distribution.png", dpi=150)
    plt.close()
    print(f"\nSaved: outputs/01_class_distribution.png")

    return counts


def show_sample_images():
    """Har class se kuch sample images ek grid mein dikhata hai."""
    print("\n" + "=" * 60)
    print("STEP 5: Sample Images per Class")
    print("=" * 60)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    for row, cls in enumerate(CLASSES):
        cls_dir = RAW_DIR / cls
        sample_paths = sorted(cls_dir.glob("*.jpg"))[:4]
        for col, img_path in enumerate(sample_paths):
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(cls, fontsize=12)
            axes[row, col].set_title(f"{cls} #{col+1}", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_sample_images.png", dpi=150)
    plt.close()
    print("Saved: outputs/02_sample_images.png")


def main():
    n_corrupted = check_corrupted_images()
    n_duplicates = check_duplicate_images()
    verify_labels()
    counts = analyze_class_distribution()
    show_sample_images()

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(f"Corrupted images removed : {n_corrupted}")
    print(f"Duplicate images removed : {n_duplicates}")
    print(f"Final class distribution : {counts}")
    print(f"Total images remaining   : {sum(counts.values())}")
    print("\n✅ Phase 1 complete. Ab 02_eda.py chalayein.")


if __name__ == "__main__":
    main()
