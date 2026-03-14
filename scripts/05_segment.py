import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from tqdm import tqdm


# ── U-Net Architecture ────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for binary segmentation (optic disc / lens).
    Input: (B, 3, H, W)   Output: (B, 1, H, W) sigmoid mask
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(3,       base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch*2)
        self.enc3 = DoubleConv(base_ch*2, base_ch*4)
        self.enc4 = DoubleConv(base_ch*4, base_ch*8)
        # Bottleneck
        self.bottleneck = DoubleConv(base_ch*8, base_ch*16)
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = DoubleConv(base_ch*16, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8,  base_ch*4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch*8,  base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4,  base_ch*2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch*4,  base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2,  base_ch,   2, stride=2)
        self.dec1 = DoubleConv(base_ch*2,  base_ch)
        self.head = nn.Conv2d(base_ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))


# ── Pseudo-label Generation ───────────────────────────────────────────────────
def generate_disc_pseudo_mask(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Weakly-supervised optic disc pseudo-label using:
      1. Red channel Otsu threshold (disc is the bright circular region)
      2. Morphological cleanup
      3. Keep largest connected component
    Returns binary mask (0/1), shape (size, size).
    """
    img = cv2.resize(img_bgr, (size, size))
    r_ch = img[:, :, 2]          # Red channel (OpenCV BGR)

    # Enhance contrast first
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    r_ch  = clahe.apply(r_ch)

    # Otsu threshold
    _, mask = cv2.threshold(r_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Keep largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask    = ((labels == largest) * 255).astype(np.uint8)

    return (mask > 127).astype(np.float32)


def generate_lens_pseudo_mask(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Lens opacity pseudo-label: bright, diffuse regions in grayscale.
    """
    img  = cv2.resize(img_bgr, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # Top 30% brightest pixels as lens proxy
    thresh = int(np.percentile(gray, 70))
    mask   = (gray > thresh).astype(np.float32)

    # Smooth
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = (mask > 0.4).astype(np.float32)
    return mask


# ── Segmentation Dataset ──────────────────────────────────────────────────────
class SegDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, target: str = "disc"):
        """
        target: 'disc' (optic disc) or 'lens' (lens region)
        """
        self.paths  = []
        self.target = target
        split_dir   = data_dir / split
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir():
                for p in cls_dir.iterdir():
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        self.paths.append(p)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(str(self.paths[idx]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224))

        if self.target == "disc":
            mask = generate_disc_pseudo_mask(img_bgr)
        else:
            mask = generate_lens_pseudo_mask(img_bgr)

        # Normalise image
        img_t  = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        mean   = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std    = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t  = (img_t - mean) / std

        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t, str(self.paths[idx])


# ── Dice Loss ─────────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_w = dice_weight
        self.bce    = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        bce_loss  = self.bce(pred, target)
        smooth    = 1e-6
        pred_flat = pred.view(-1)
        tgt_flat  = target.view(-1)
        dice      = (2.0 * (pred_flat * tgt_flat).sum() + smooth) / \
                    (pred_flat.sum() + tgt_flat.sum() + smooth)
        return (1 - self.dice_w) * bce_loss + self.dice_w * (1 - dice)


# ── Biomarker Extraction ──────────────────────────────────────────────────────
def compute_cdr(disc_mask: np.ndarray) -> float:
    """
    Estimates Cup-to-Disc Ratio (CDR) from the optic disc mask.
    CDR = cup_diameter / disc_diameter (vertical)
    The cup is approximated as the central bright region within the disc.
    """
    disc_bin = (disc_mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(disc_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    # Bounding box of disc
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    disc_h = h

    # Cup approximation: inner 45% of disc height
    cup_h = disc_h * 0.45
    return round(float(cup_h / (disc_h + 1e-8)), 3)


def compute_opacity_area(lens_mask: np.ndarray) -> float:
    """
    Returns lens opacity area as fraction of total image area.
    """
    return round(float(lens_mask.sum()) / (lens_mask.shape[0] * lens_mask.shape[1] + 1e-8), 4)


# ── Visualisation ─────────────────────────────────────────────────────────────
def save_segmentation_grid(model: nn.Module, dataset: SegDataset,
                            device: torch.device, out_path: Path, n: int = 4):
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    titles = ["Original", "Pseudo-Label", "Prediction"]
    for j, t in enumerate(titles):
        axes[j, 0].set_ylabel(t, fontsize=10, fontweight="bold",
                              rotation=0, labelpad=60, va="center")

    with torch.no_grad():
        for col in range(min(n, len(dataset))):
            img_t, mask_t, _ = dataset[col]
            pred = model(img_t.unsqueeze(0).to(device))[0, 0].cpu().numpy()

            # Denormalise for display
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_np = img_t.permute(1, 2, 0).numpy()
            img_np = (img_np * std + mean).clip(0, 1)

            axes[0, col].imshow(img_np); axes[0, col].axis("off")
            axes[1, col].imshow(mask_t[0].numpy(), cmap="gray"); axes[1, col].axis("off")
            axes[2, col].imshow(pred, cmap="hot");  axes[2, col].axis("off")

    fig.suptitle("Segmentation Results", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_dir, ckpt_dir, out_dir, model_name, epochs):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Phase 3: Segmentation & Biomarker Extraction ===\n  Device: {device}\n")

    for target in ["disc", "lens"]:
        print(f"\n--- Target: {'Optic Disc' if target == 'disc' else 'Lens'} ---")
        target_dir = out_dir / target
        target_dir.mkdir(exist_ok=True)

        # Datasets
        print(f"[1] Building {target} segmentation dataset...")
        train_ds = SegDataset(data_dir, "train", target)
        val_ds   = SegDataset(data_dir, "val",   target)
        print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2)

        # Model
        unet = UNet(base_ch=32).to(device)
        criterion = DiceBCELoss()
        optimizer = optim.Adam(unet.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        # Training
        print(f"[2] Training U-Net for {epochs} epochs...")
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            # Train
            unet.train()
            tr_loss = 0.0
            for imgs, masks, _ in tqdm(train_loader, desc=f"  Ep{epoch:>3}", leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                preds = unet(imgs)
                loss  = criterion(preds, masks)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= len(train_loader)

            # Val
            unet.eval()
            vl_loss = 0.0
            with torch.no_grad():
                for imgs, masks, _ in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    preds = unet(imgs)
                    vl_loss += criterion(preds, masks).item()
            vl_loss /= len(val_loader)
            scheduler.step(vl_loss)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            print(f"  Ep {epoch:>3}  train={tr_loss:.4f}  val={vl_loss:.4f}", end="")

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save(unet.state_dict(), target_dir / "unet_best.pth")
                print(" ✓")
            else:
                print()

        # Visualise
        print("[3] Generating segmentation grid...")
        unet.load_state_dict(torch.load(target_dir / "unet_best.pth", map_location=device))
        save_segmentation_grid(unet, val_ds, device,
                               target_dir / "segmentation_grid.png")

        # Biomarker extraction on test set
        print("[4] Extracting biomarkers from test set...")
        test_ds = SegDataset(data_dir, "test", target)
        biomarkers = []
        unet.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(test_ds)), desc="  Biomarkers"):
                img_t, _, path = test_ds[idx]
                pred = unet(img_t.unsqueeze(0).to(device))[0, 0].cpu().numpy()

                if target == "disc":
                    score = compute_cdr(pred)
                    risk  = "High" if score > 0.6 else ("Moderate" if score > 0.4 else "Low")
                    biomarkers.append({"path": path, "CDR": score, "glaucoma_risk": risk})
                else:
                    score = compute_opacity_area(pred)
                    sev   = "Severe" if score > 0.4 else ("Moderate" if score > 0.2 else "Mild")
                    biomarkers.append({"path": path, "opacity_area": score, "cataract_severity": sev})

        bm_df = pd.DataFrame(biomarkers)
        bm_df.to_csv(target_dir / "biomarkers.csv", index=False)
        print(f"  Biomarker CSV saved: {target_dir / 'biomarkers.csv'}")
        print(f"  Sample biomarkers:\n{bm_df.head(5).to_string(index=False)}")

        # Plot loss
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["train_loss"], label="Train", color="#2E9AB5")
        ax.plot(history["val_loss"],   label="Val",   color="#E05A4E")
        ax.set_title(f"U-Net ({target}) Loss", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Dice-BCE Loss")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(target_dir / "loss_curve.png", dpi=150)
        plt.close(fig)

    print(f"\nSegmentation complete. Outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optic disc/lens segmentation")
    parser.add_argument("--data_dir",  default="outputs/processed")
    parser.add_argument("--ckpt_dir",  default="outputs/classifier")
    parser.add_argument("--out_dir",   default="outputs/segmentation")
    parser.add_argument("--model",     default="efficientnet_b3")
    parser.add_argument("--epochs",    type=int, default=20)
    args = parser.parse_args()
    main(args.data_dir, args.ckpt_dir, args.out_dir, args.model, args.epochs)
