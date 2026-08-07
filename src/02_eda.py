"""
02_eda.py
---------
PHASE 2 — Exploratory Data Analysis (EDA)

Ye script har class ki images analyze karta hai:
1. Image resolution statistics
2. RGB channel distribution
3. Brightness analysis
4. Image file size analysis
5. Visualizations: bar chart, pie chart, histogram, box plot
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
CLASSES = ["cardboard", "glass", "paper"]
COLORS = {"cardboard": "#C9A24B", "glass": "#5DBEA3", "paper": "#7C8FA6"}


def collect_stats():
    """Har image se resolution, RGB means, brightness, aur file size collect karta hai."""
    print("=" * 60)
    print("Collecting image statistics...")
    print("=" * 60)

    records = []
    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        for img_path in cls_dir.glob("*.jpg"):
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                arr = np.array(img_rgb)

                width, height = img.size
                file_size_kb = img_path.stat().st_size / 1024

                r_mean = arr[:, :, 0].mean()
                g_mean = arr[:, :, 1].mean()
                b_mean = arr[:, :, 2].mean()
                brightness = arr.mean()  # Overall brightness (avg of all channels)

                records.append({
                    "class": cls,
                    "width": width,
                    "height": height,
                    "file_size_kb": file_size_kb,
                    "r_mean": r_mean,
                    "g_mean": g_mean,
                    "b_mean": b_mean,
                    "brightness": brightness,
                })

    print(f"Collected stats for {len(records)} images")
    return records


def resolution_stats(records):
    print("\n" + "=" * 60)
    print("STEP 1: Image Resolution Statistics")
    print("=" * 60)
    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]
    print(f"  Width  -> min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.1f}")
    print(f"  Height -> min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.1f}")
    print(f"  Most common resolution: {max(set(zip(widths, heights)), key=lambda x: (widths.count(x[0]) if True else 0))}")


def plot_brightness_histogram(records):
    """Har class ki brightness distribution ka histogram."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cls in CLASSES:
        vals = [r["brightness"] for r in records if r["class"] == cls]
        ax.hist(vals, bins=25, alpha=0.6, label=cls, color=COLORS[cls])
    ax.set_title("Brightness Distribution by Class (Histogram)")
    ax.set_xlabel("Average Brightness (0-255)")
    ax.set_ylabel("Number of Images")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_brightness_histogram.png", dpi=150)
    plt.close()
    print("Saved: outputs/03_brightness_histogram.png")


def plot_filesize_boxplot(records):
    """Har class ki image file size ka box plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [[r["file_size_kb"] for r in records if r["class"] == cls] for cls in CLASSES]
    bp = ax.boxplot(data, labels=CLASSES, patch_artist=True)
    for patch, cls in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(COLORS[cls])
        patch.set_alpha(0.7)
    ax.set_title("Image File Size by Class (Box Plot)")
    ax.set_ylabel("File Size (KB)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_filesize_boxplot.png", dpi=150)
    plt.close()
    print("Saved: outputs/04_filesize_boxplot.png")


def plot_rgb_distribution(records):
    """Har class ke average RGB channel values ka bar chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASSES))
    width = 0.25

    r_means = [np.mean([r["r_mean"] for r in records if r["class"] == c]) for c in CLASSES]
    g_means = [np.mean([r["g_mean"] for r in records if r["class"] == c]) for c in CLASSES]
    b_means = [np.mean([r["b_mean"] for r in records if r["class"] == c]) for c in CLASSES]

    ax.bar(x - width, r_means, width, label="Red channel", color="#D9534F")
    ax.bar(x, g_means, width, label="Green channel", color="#5CB85C")
    ax.bar(x + width, b_means, width, label="Blue channel", color="#5BC0DE")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Average Channel Intensity (0-255)")
    ax.set_title("RGB Channel Distribution by Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_rgb_distribution.png", dpi=150)
    plt.close()
    print("Saved: outputs/05_rgb_distribution.png")


def plot_images_per_class(records):
    """Bar + pie chart - kitni images har class mein hain (EDA confirm)."""
    counts = {cls: sum(1 for r in records if r["class"] == cls) for cls in CLASSES}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(counts.keys(), counts.values(), color=[COLORS[c] for c in CLASSES])
    axes[0].set_title("Images per Class")
    axes[0].set_ylabel("Count")

    axes[1].pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%",
                colors=[COLORS[c] for c in CLASSES])
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_images_per_class_eda.png", dpi=150)
    plt.close()
    print("Saved: outputs/06_images_per_class_eda.png")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    records = collect_stats()

    resolution_stats(records)

    print("\n" + "=" * 60)
    print("STEP 2: Generating Visualizations")
    print("=" * 60)
    plot_images_per_class(records)
    plot_brightness_histogram(records)
    plot_filesize_boxplot(records)
    plot_rgb_distribution(records)

    print("\n✅ Phase 2 (EDA) complete. Ab 03_preprocessing_augmentation.py chalayein.")


if __name__ == "__main__":
    main()
