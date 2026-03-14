import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# ── cDCGAN Components ─────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Conditional Generator.
    Input: (z, class_label)  →  (3, 64, 64) RGB image
    """
    def __init__(self, z_dim: int, n_classes: int, base_ch: int = 64):
        super().__init__()
        self.z_dim     = z_dim
        self.n_classes = n_classes
        self.embed     = nn.Embedding(n_classes, n_classes)

        in_dim = z_dim + n_classes
        self.net = nn.Sequential(
            # in_dim → base_ch*8 × 4 × 4
            nn.ConvTranspose2d(in_dim, base_ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_ch * 8), nn.ReLU(True),
            # 4 → 8
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4), nn.ReLU(True),
            # 8 → 16
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(True),
            # 16 → 32
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(True),
            # 32 → 64
            nn.ConvTranspose2d(base_ch, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb  = self.embed(labels)                  # (B, n_classes)
        inp  = torch.cat([z, emb], dim=1)          # (B, z_dim + n_classes)
        inp  = inp.unsqueeze(-1).unsqueeze(-1)      # (B, C, 1, 1)
        return self.net(inp)


class Discriminator(nn.Module):
    """
    AC-GAN Discriminator.
    Input: (3, 64, 64) image → (real/fake probability, class logits)
    """
    def __init__(self, n_classes: int, base_ch: int = 64):
        super().__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(
            # 64 → 32
            nn.Conv2d(3, base_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 32 → 16
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2), nn.LeakyReLU(0.2, True),
            # 16 → 8
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4), nn.LeakyReLU(0.2, True),
            # 8 → 4
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 8), nn.LeakyReLU(0.2, True),
        )
        self.adv_head = nn.Sequential(
            nn.Conv2d(base_ch * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(base_ch * 8, n_classes, 4, 1, 0, bias=False)
        )

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        adv  = self.adv_head(feat).view(-1)
        cls  = self.cls_head(feat).view(-1, self.n_classes)
        return adv, cls


# ── Training Dataset ──────────────────────────────────────────────────────────
class GanDataset(Dataset):
    def __init__(self, data_dir: Path, split: str = "train", img_size: int = 64):
        self.paths, self.labels = [], []
        self.img_size = img_size

        split_dir = data_dir / split
        classes   = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        for cls in classes:
            for p in sorted((split_dir / cls).iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.paths.append(p)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = (img.astype(np.float32) / 127.5) - 1.0   # [-1, 1]
        return torch.from_numpy(img.transpose(2, 0, 1)), self.labels[idx]


# ── Weights init ──────────────────────────────────────────────────────────────
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ── Image Grid ────────────────────────────────────────────────────────────────
def save_image_grid(gen: nn.Module, z_dim: int, n_classes: int,
                    device: torch.device, out_path: Path, n_per_class: int = 4):
    gen.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(n_classes, n_per_class,
                                 figsize=(n_per_class * 2, n_classes * 2))
        for row in range(n_classes):
            z      = torch.randn(n_per_class, z_dim, device=device)
            labels = torch.full((n_per_class,), row, dtype=torch.long, device=device)
            imgs   = gen(z, labels)
            for col in range(n_per_class):
                img = imgs[col].cpu().permute(1, 2, 0).numpy()
                img = ((img + 1) / 2).clip(0, 1)
                axes[row, col].imshow(img)
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(f"Class {row}", fontsize=9,
                                    rotation=0, labelpad=45, va="center")
        fig.suptitle("Generated Fundus Images (per class)", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Synthetic Image Export ─────────────────────────────────────────────────────
def export_synthetic_images(gen: nn.Module, z_dim: int,
                             idx_to_class: dict, out_dir: Path,
                             n_per_class: int, device: torch.device):
    gen.eval()
    counts = {}
    with torch.no_grad():
        for cls_idx, cls_name in idx_to_class.items():
            save_dir = out_dir / cls_name
            save_dir.mkdir(parents=True, exist_ok=True)
            n = n_per_class
            z      = torch.randn(n, z_dim, device=device)
            labels = torch.full((n,), int(cls_idx), dtype=torch.long, device=device)
            imgs   = gen(z, labels)
            for i, img in enumerate(imgs):
                img_np = img.cpu().permute(1, 2, 0).numpy()
                img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                # Upsample to 224x224 for classifier compatibility
                img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(save_dir / f"syn_{i:04d}.png"), img_bgr)
            counts[cls_name] = n
    return counts


# ── Main Training Loop ────────────────────────────────────────────────────────
def main(data_dir, out_dir, epochs, batch_size, n_gen, z_dim):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Phase 4: Conditional GAN Augmentation ===\n  Device: {device}\n")

    print("[1/5] Building dataset...")
    dataset = GanDataset(data_dir, "train", img_size=64)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, drop_last=True)
    n_classes   = len(dataset.class_to_idx)
    idx_to_class = dataset.idx_to_class
    print(f"  Images: {len(dataset)}  |  Classes: {n_classes}")

    print("[2/5] Initialising cDCGAN...")
    G = Generator(z_dim=z_dim, n_classes=n_classes).to(device)
    D = Discriminator(n_classes=n_classes).to(device)
    G.apply(weights_init); D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    adv_criterion = nn.BCELoss()
    cls_criterion = nn.CrossEntropyLoss()

    history = {"G_loss": [], "D_loss": [], "D_real_acc": [], "D_fake_acc": []}

    print(f"[3/5] Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        G_losses, D_losses = [], []
        d_real_correct, d_fake_correct, n_total = 0, 0, 0

        for real_imgs, real_labels in loader:
            B = real_imgs.size(0)
            real_imgs   = real_imgs.to(device)
            real_labels = real_labels.to(device)
            real_tgt    = torch.ones(B, device=device)
            fake_tgt    = torch.zeros(B, device=device)

            # ── Discriminator step ──
            opt_D.zero_grad()

            # Real
            adv_real, cls_real = D(real_imgs)
            d_loss_real = adv_criterion(adv_real, real_tgt) + \
                          cls_criterion(cls_real, real_labels)

            # Fake
            z          = torch.randn(B, z_dim, device=device)
            fake_labels = torch.randint(0, n_classes, (B,), device=device)
            fake_imgs  = G(z, fake_labels).detach()
            adv_fake, cls_fake = D(fake_imgs)
            d_loss_fake = adv_criterion(adv_fake, fake_tgt) + \
                          cls_criterion(cls_fake, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            opt_D.step()

            D_losses.append(d_loss.item())
            d_real_correct += (adv_real > 0.5).sum().item()
            d_fake_correct += (adv_fake < 0.5).sum().item()
            n_total        += B

            # ── Generator step ──
            opt_G.zero_grad()
            z          = torch.randn(B, z_dim, device=device)
            fake_labels = torch.randint(0, n_classes, (B,), device=device)
            fake_imgs  = G(z, fake_labels)
            adv_fake, cls_fake = D(fake_imgs)

            g_loss = adv_criterion(adv_fake, real_tgt) + \
                     cls_criterion(cls_fake, fake_labels)
            g_loss.backward()
            opt_G.step()
            G_losses.append(g_loss.item())

        ep_g = np.mean(G_losses)
        ep_d = np.mean(D_losses)
        ep_dr = d_real_correct / n_total
        ep_df = d_fake_correct / n_total
        history["G_loss"].append(ep_g)
        history["D_loss"].append(ep_d)
        history["D_real_acc"].append(ep_dr)
        history["D_fake_acc"].append(ep_df)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:>4}/{epochs}  G={ep_g:.4f}  D={ep_d:.4f}  "
                  f"D_real={ep_dr:.3f}  D_fake={ep_df:.3f}")

    # Save model
    torch.save(G.state_dict(), out_dir / "generator.pth")
    torch.save(D.state_dict(), out_dir / "discriminator.pth")
    with open(out_dir / "gan_config.json", "w") as f:
        json.dump({"z_dim": z_dim, "n_classes": n_classes,
                   "idx_to_class": {str(k): v for k, v in idx_to_class.items()}}, f, indent=2)
    print("  Models saved.")

    # Sample grid
    print("[4/5] Generating sample image grid...")
    save_image_grid(G, z_dim, n_classes, device,
                    out_dir / "generated_samples.png", n_per_class=4)

    # Export synthetic images
    print(f"[5/5] Exporting {n_gen} synthetic images per class...")
    counts = export_synthetic_images(G, z_dim, idx_to_class,
                                     out_dir / "images", n_gen, device)
    print(f"  Exported: {counts}")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_x = range(1, len(history["G_loss"]) + 1)
    ax1.plot(epochs_x, history["G_loss"], label="G Loss", color="#2E9AB5")
    ax1.plot(epochs_x, history["D_loss"], label="D Loss", color="#E05A4E")
    ax1.set_title("GAN Training Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.legend(); ax1.spines[["top","right"]].set_visible(False)

    ax2.plot(epochs_x, history["D_real_acc"], label="D Real Acc", color="#5BAD6F")
    ax2.plot(epochs_x, history["D_fake_acc"], label="D Fake Acc", color="#F4A642")
    ax2.set_title("Discriminator Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.legend(); ax2.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "gan_training_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nGAN augmentation complete.")
    print(f"  Synthetic images saved to: {out_dir / 'images'}")
    print(f"  To use in training, add --augment_dir {out_dir / 'images'} to 03_classify.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN augmentation")
    parser.add_argument("--data_dir",   default="outputs/processed")
    parser.add_argument("--out_dir",    default="outputs/augmented")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--n_gen",      type=int,   default=50,
                        help="Synthetic images to generate per class")
    parser.add_argument("--z_dim",      type=int,   default=128)
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.epochs,
         args.batch_size, args.n_gen, args.z_dim)
