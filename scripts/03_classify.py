import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt


# ── Dataset ───────────────────────────────────────────────────────────────────
class RetinalDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, img_size: int = 224,
                 augment: bool = False):
        self.paths, self.labels, self.class_to_idx = [], [], {}
        self.augment = augment
        self.img_size = img_size

        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        for cls in classes:
            for img_path in sorted((split_dir / cls).iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls])

        # Augmentation pipeline (training only)
        self.train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        self.val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment:
            img = self.train_transform(image=img)["image"]
        else:
            img = self.val_transform(image=img)["image"]
        return img, self.labels[idx]


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Loads a pretrained backbone via timm and replaces the classifier head.
    Supported: efficientnet_b3, resnet50, vit_base_patch16_224
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    print(f"  Model: {model_name}  |  Params: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ── Training helpers ──────────────────────────────────────────────────────────
def get_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    weights = compute_class_weight("balanced",
                                   classes=np.arange(num_classes),
                                   y=np.array(labels))
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds      = outputs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, model_name: str, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", color="#2E9AB5")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#E05A4E")
    ax1.set_title(f"{model_name} — Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(epochs, history["train_acc"], label="Train", color="#2E9AB5")
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="#E05A4E")
    ax2.set_title(f"{model_name} — Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.spines[["top", "right"]].set_visible(False)

    # Mark best epoch
    best_ep = np.argmin(history["val_loss"]) + 1
    for ax in [ax1, ax2]:
        ax.axvline(best_ep, ls="--", color="grey", alpha=0.5, label=f"Best (ep {best_ep})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_dir, out_dir, model_name, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Phase 2: Classification Training ===")
    print(f"  Device: {device}  |  Model: {model_name}\n")

    # Datasets & loaders
    print("[1/5] Loading datasets...")
    train_ds = RetinalDataset(data_dir, "train", augment=True)
    val_ds   = RetinalDataset(data_dir, "val",   augment=False)
    test_ds  = RetinalDataset(data_dir, "test",  augment=False)

    num_classes = len(train_ds.class_to_idx)
    print(f"  Classes ({num_classes}): {train_ds.class_to_idx}")
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

    # Save class map
    with open(out_dir / "class_map.json", "w") as f:
        json.dump(train_ds.idx_to_class, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Model
    print("[2/5] Building model...")
    model = build_model(model_name, num_classes).to(device)

    # Class-weighted loss
    class_weights = get_class_weights(train_ds.labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    print(f"[3/5] Training for {epochs} epochs...")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss  = float("inf")
    patience_count = 0
    patience       = 8

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"  Ep {epoch:>3}/{epochs}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.3f}/{vl_acc:.3f}  "
              f"({elapsed:.1f}s)", end="")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_count = 0
            ckpt_path = out_dir / f"{model_name}_best.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "class_map": train_ds.idx_to_class}, ckpt_path)
            print(" ✓ saved", end="")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break
        print()

    # Plot curves
    print("[4/5] Saving training curves...")
    plot_training_curves(history, model_name, out_dir / f"{model_name}_curves.png")
    pd.DataFrame(history).to_csv(out_dir / f"{model_name}_history.csv", index=False)

    # Final test evaluation
    print("[5/5] Evaluating on test set...")
    ckpt = torch.load(out_dir / f"{model_name}_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, preds, labels = eval_epoch(model, test_loader, criterion, device)

    print(f"\n  Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

    # Save test predictions
    pred_df = pd.DataFrame({"true": labels, "pred": preds})
    pred_df.to_csv(out_dir / f"{model_name}_test_predictions.csv", index=False)

    # Save training summary
    summary = {
        "model":        model_name,
        "best_val_loss": best_val_loss,
        "test_accuracy": test_acc,
        "epochs_run":    len(history["train_loss"]),
        "num_classes":   num_classes,
    }
    with open(out_dir / f"{model_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nClassifier training complete. Outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cataract classifier")
    parser.add_argument("--data_dir",   default="outputs/processed")
    parser.add_argument("--out_dir",    default="outputs/classifier")
    parser.add_argument("--model",      default="efficientnet_b3",
                        choices=["efficientnet_b3", "resnet50", "vit_base_patch16_224"])
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.model,
         args.epochs, args.batch_size, args.lr, args.seed)
