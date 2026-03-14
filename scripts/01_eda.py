import argparse
import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_DIRS = {
    "Normal":         "1_normal",
    "Cataract":       "2_cataract",
    "Glaucoma":       "2_glaucoma",
    "Retinal Disease": "3_retina_disease",
}
CLASS_COLORS = ["#2E9AB5", "#E05A4E", "#F4A642", "#5BAD6F"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def collect_image_paths(data_dir: Path) -> dict[str, list[Path]]:
    paths = {}
    for label, folder in CLASS_DIRS.items():
        d = data_dir / folder
        if not d.exists():
            print(f"  [WARN] Missing folder: {d}")
            paths[label] = []
        else:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            paths[label] = [p for p in sorted(d.iterdir()) if p.suffix.lower() in exts]
    return paths


def image_stats(path: Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "path":       str(path),
        "width":      w,
        "height":     h,
        "aspect":     round(w / h, 3),
        "brightness": round(float(np.mean(gray)), 2),
        "contrast":   round(float(np.std(gray)), 2),
    }


# ── Plot functions ─────────────────────────────────────────────────────────────
def plot_class_distribution(counts: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.keys(), counts.values(), color=CLASS_COLORS, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_ylim(0, max(counts.values()) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_sample_grid(class_paths: dict, out_path: Path, n_samples: int = 4):
    n_classes = len(class_paths)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 3, n_classes * 3))
    fig.suptitle("Sample Images per Class", fontsize=15, fontweight="bold", y=1.01)

    for row, (label, paths) in enumerate(class_paths.items()):
        chosen = paths[:n_samples] if len(paths) >= n_samples else paths
        for col in range(n_samples):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(chosen):
                img = cv2.imread(str(chosen[col]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                ax.imshow(img)
            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold", rotation=0,
                              labelpad=60, va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_resolution_distribution(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    palette = {label: col for label, col in zip(CLASS_DIRS.keys(), CLASS_COLORS)}

    for ax, col, title in zip(axes,
                               ["width", "height", "aspect"],
                               ["Image Width (px)", "Image Height (px)", "Aspect Ratio (W/H)"]):
        for label, grp in df.groupby("class"):
            ax.hist(grp[col], bins=20, alpha=0.6, label=label, color=palette[label])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_brightness_contrast(df: pd.DataFrame, out_path: Path):
    palette = {label: col for label, col in zip(CLASS_DIRS.keys(), CLASS_COLORS)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, title in zip(axes,
                                  ["brightness", "contrast"],
                                  ["Mean Brightness", "Contrast (Std Dev)"]):
        for label, grp in df.groupby("class"):
            ax.hist(grp[metric], bins=25, alpha=0.65, label=label, color=palette[label])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path):
    num_cols = ["width", "height", "aspect", "brightness", "contrast"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_dir: str, out_dir: str):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Phase 1: Exploratory Data Analysis ===\n")

    # 1. Collect paths
    print("[1/6] Collecting image paths...")
    class_paths = collect_image_paths(data_dir)
    counts = {label: len(paths) for label, paths in class_paths.items()}

    print("\n  Class Counts:")
    total = 0
    for label, n in counts.items():
        print(f"    {label:<20} {n:>4} images")
        total += n
    print(f"    {'TOTAL':<20} {total:>4} images")

    # 2. Class distribution plot
    print("\n[2/6] Plotting class distribution...")
    plot_class_distribution(counts, out_dir / "01_class_distribution.png")

    # 3. Sample grid
    print("[3/6] Generating sample image grid...")
    plot_sample_grid(class_paths, out_dir / "02_sample_grid.png")

    # 4. Per-image statistics
    print("[4/6] Computing per-image statistics...")
    records = []
    for label, paths in class_paths.items():
        for p in tqdm(paths, desc=f"  {label}", leave=False):
            stats = image_stats(p)
            if stats:
                stats["class"] = label
                records.append(stats)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "image_stats.csv", index=False)
    print(f"  Saved stats CSV: {out_dir / 'image_stats.csv'}")

    # 5. Distribution plots
    print("[5/6] Plotting resolution & intensity distributions...")
    plot_resolution_distribution(df, out_dir / "03_resolution_distribution.png")
    plot_brightness_contrast(df,    out_dir / "04_brightness_contrast.png")
    plot_correlation_heatmap(df,    out_dir / "05_correlation_heatmap.png")

    # 6. Summary report
    print("[6/6] Generating summary report...")
    summary = df.groupby("class")[["width", "height", "brightness", "contrast"]].agg(
        ["mean", "std", "min", "max"]
    ).round(2)
    summary.to_csv(out_dir / "summary_report.csv")

    # Console print
    print("\n  Per-Class Summary Statistics:")
    print(df.groupby("class")[["width", "height", "brightness", "contrast"]]
          .mean().round(1).to_string())

    # Imbalance warning
    max_n, min_n = max(counts.values()), min(counts.values())
    ratio = max_n / max(min_n, 1)
    if ratio > 1.5:
        print(f"\n  [WARN] Class imbalance detected — ratio {ratio:.1f}x. "
              "Consider weighted loss or oversampling in training.")

    print(f"\nEDA complete. All outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cataract Dataset EDA")
    parser.add_argument("--data_dir", default="data/dataset",
                        help="Path to the dataset root folder")
    parser.add_argument("--out_dir",  default="outputs/eda",
                        help="Directory to save EDA outputs")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
