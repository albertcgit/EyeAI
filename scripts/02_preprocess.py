import argparse
import os
import shutil
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_DIRS = {
    "normal":         "1_normal",
    "cataract":       "2_cataract",
    "glaucoma":       "2_glaucoma",
    "retinal_disease": "3_retina_disease",
}
CLASS_TO_IDX = {k: i for i, k in enumerate(CLASS_DIRS.keys())}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(path: Path, img_size: int) -> np.ndarray:
    """
    Full preprocessing pipeline for a single retinal fundus image:
      1. Load in BGR
      2. Resize to img_size x img_size
      3. Convert to LAB colour space
      4. Apply CLAHE to L channel (enhances local contrast, standard for fundus)
      5. Convert back to BGR
      6. Normalise to [0, 255] uint8
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    # Centre-crop to square first to preserve aspect ratio
    h, w = img.shape[:2]
    side = min(h, w)
    top  = (h - side) // 2
    left = (w - side) // 2
    img  = img[top:top+side, left:left+side]

    # Resize
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

    # CLAHE on L channel
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    img   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


# ── Split & Save ──────────────────────────────────────────────────────────────
def build_file_list(data_dir: Path) -> pd.DataFrame:
    records = []
    for label, folder in CLASS_DIRS.items():
        d = data_dir / folder
        if not d.exists():
            print(f"  [WARN] Folder not found: {d}")
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                records.append({"path": str(p), "label": label,
                                 "class_idx": CLASS_TO_IDX[label]})
    return pd.DataFrame(records)


def split_dataframe(df: pd.DataFrame, seed: int):
    """Stratified 70 / 15 / 15 split."""
    train_df, tmp_df = train_test_split(df, test_size=0.30,
                                        stratify=df["label"], random_state=seed)
    val_df, test_df  = train_test_split(tmp_df, test_size=0.50,
                                        stratify=tmp_df["label"], random_state=seed)
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"]   = "val"
    test_df  = test_df.copy();  test_df["split"]  = "test"
    return train_df, val_df, test_df


def save_split(rows: pd.DataFrame, split_name: str,
               out_dir: Path, img_size: int) -> list[dict]:
    """Preprocesses and saves images for one split, returns manifest rows."""
    manifest = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"  {split_name}"):
        src  = Path(row["path"])
        dest_dir = out_dir / split_name / row["label"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        try:
            img = preprocess_image(src, img_size)
            cv2.imwrite(str(dest), img)
            manifest.append({
                "split": split_name,
                "label": row["label"],
                "class_idx": row["class_idx"],
                "original_path": str(src),
                "processed_path": str(dest),
            })
        except Exception as e:
            print(f"\n  [WARN] Skipping {src.name}: {e}")

    return manifest


def plot_split_distribution(manifest_df: pd.DataFrame, out_path: Path):
    splits = ["train", "val", "test"]
    classes = list(CLASS_DIRS.keys())
    colors  = ["#2E9AB5", "#E05A4E", "#F4A642", "#5BAD6F"]

    x = np.arange(len(splits))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (cls, col) in enumerate(zip(classes, colors)):
        counts = [len(manifest_df[(manifest_df.split == s) & (manifest_df.label == cls)])
                  for s in splits]
        bars = ax.bar(x + i * width, counts, width, label=cls, color=col, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        str(int(h)), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s.capitalize() for s in splits], fontsize=11)
    ax.set_title("Images per Class per Split", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved split chart: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_dir: str, out_dir: str, img_size: int, seed: int):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Phase 1: Preprocessing & Dataset Split ===\n")

    # Build file list
    print("[1/4] Scanning raw dataset...")
    df = build_file_list(data_dir)
    print(f"  Found {len(df)} images across {df['label'].nunique()} classes")

    if df.empty:
        print("  [ERROR] No images found. Check --data_dir path.")
        return

    # Split
    print("[2/4] Splitting into train / val / test (70/15/15)...")
    train_df, val_df, test_df = split_dataframe(df, seed)
    print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # Preprocess & save
    print("[3/4] Preprocessing and saving images...")
    manifest = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        manifest.extend(save_split(split_df, split_name, out_dir, img_size))

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out_dir / "manifest.csv", index=False)
    print(f"\n  Manifest saved: {out_dir / 'manifest.csv'}")

    # Save class map
    class_map = {v: k for k, v in CLASS_TO_IDX.items()}
    with open(out_dir / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"  Class map saved: {out_dir / 'class_map.json'}")

    # Save config
    config = {"img_size": img_size, "seed": seed,
              "classes": list(CLASS_DIRS.keys()),
              "class_to_idx": CLASS_TO_IDX}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Plot
    print("[4/4] Plotting split distribution...")
    plot_split_distribution(manifest_df, out_dir / "split_distribution.png")

    # Final summary
    print("\n  Split Summary:")
    summary = manifest_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(summary.to_string())

    print(f"\nPreprocessing complete. Processed images saved to: {out_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split cataract dataset")
    parser.add_argument("--data_dir", default="data/dataset")
    parser.add_argument("--out_dir",  default="outputs/processed")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.img_size, args.seed)
