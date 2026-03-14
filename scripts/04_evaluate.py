import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    cohen_kappa_score, f1_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from tqdm import tqdm

# Reuse dataset class from 03_classify
import sys
sys.path.insert(0, str(Path(__file__).parent))
from classify_03 import RetinalDataset   # fallback import alias

# If standalone, redefine minimal dataset
try:
    from classify_03 import RetinalDataset
except ImportError:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    class RetinalDataset:
        def __init__(self, data_dir, split, augment=False):
            from torch.utils.data import Dataset
            import cv2, albumentations as A
            from albumentations.pytorch import ToTensorV2

            class _DS(Dataset):
                def __init__(self, data_dir, split):
                    self.paths, self.labels = [], []
                    split_dir = Path(data_dir) / split
                    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
                    self.class_to_idx = {c: i for i, c in enumerate(classes)}
                    self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
                    for cls in classes:
                        for p in sorted((split_dir / cls).iterdir()):
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                                self.paths.append(p)
                                self.labels.append(self.class_to_idx[cls])
                    self.tf = A.Compose([
                        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                        ToTensorV2(),
                    ])

                def __len__(self): return len(self.paths)
                def __getitem__(self, idx):
                    img = cv2.imread(str(self.paths[idx]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return self.tf(image=img)["image"], self.labels[idx]

            self._ds = _DS(data_dir, split)
            for attr in ["paths", "labels", "class_to_idx", "idx_to_class"]:
                setattr(self, attr, getattr(self._ds, attr))
            self.__len__ = self._ds.__len__
            self.__getitem__ = self._ds.__getitem__


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """Minimal Grad-CAM implementation for CNN models."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients, self.activations = None, None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _, __, output):
        self.activations = output.detach()

    def _save_gradients(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score  = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def get_gradcam_layer(model, model_name: str):
    """Returns the last conv feature layer for Grad-CAM."""
    if "efficientnet" in model_name:
        return model.conv_head
    elif "resnet" in model_name:
        return model.layer4[-1]
    elif "vit" in model_name:
        return model.blocks[-1].norm1
    else:
        # Generic fallback: last conv layer
        for layer in reversed(list(model.modules())):
            if isinstance(layer, nn.Conv2d):
                return layer
    raise ValueError(f"Cannot determine Grad-CAM layer for {model_name}")


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_score: np.ndarray,
                    class_names: list[str], out_path: Path):
    colors = ["#2E9AB5", "#E05A4E", "#F4A642", "#5BAD6F"]
    n_classes = len(class_names)
    y_bin  = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gradcam_grid(model, gradcam, dataset, class_names: list[str],
                      device, n_per_class: int, out_path: Path):
    """Generates Grad-CAM overlays for n_per_class samples from each class."""
    n_classes = len(class_names)
    fig = plt.figure(figsize=(n_per_class * 3.5, n_classes * 3.5))
    fig.suptitle("Grad-CAM Visualisations", fontsize=14, fontweight="bold")

    model.eval()
    class_samples = {i: [] for i in range(n_classes)}

    for idx in range(len(dataset)):
        lbl = dataset.labels[idx]
        if len(class_samples[lbl]) < n_per_class:
            class_samples[lbl].append(idx)
        if all(len(v) >= n_per_class for v in class_samples.values()):
            break

    for row, cls_idx in enumerate(range(n_classes)):
        for col, sample_idx in enumerate(class_samples[cls_idx]):
            ax = fig.add_subplot(n_classes, n_per_class, row * n_per_class + col + 1)

            # Load original image
            img_bgr = cv2.imread(str(dataset.paths[sample_idx]))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (224, 224))

            # Get tensor
            img_tensor, _ = dataset[sample_idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Grad-CAM
            cam = gradcam.generate(img_tensor, cls_idx)
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8),
                                        cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = (img_rgb * 0.6 + heatmap * 0.4).clip(0, 255).astype(np.uint8)

            ax.imshow(overlay)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(class_names[cls_idx], fontsize=10,
                              fontweight="bold", rotation=0, labelpad=60, va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grad-CAM grid saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_dir, ckpt_dir, out_dir, model_name, n_gradcam):
    data_dir = Path(data_dir)
    ckpt_dir = Path(ckpt_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Phase 2: Model Evaluation ===\n  Device: {device}\n")

    # Load test dataset
    print("[1/5] Loading test dataset...")
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import Dataset

    class SimpleDataset(Dataset):
        def __init__(self, data_dir, split):
            self.paths, self.labels = [], []
            split_dir = Path(data_dir) / split
            classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            self.class_names  = classes
            for cls in classes:
                for p in sorted((split_dir / cls).iterdir()):
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        self.paths.append(p)
                        self.labels.append(self.class_to_idx[cls])
            self.tf = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            img = cv2.imread(str(self.paths[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.tf(image=img)["image"], self.labels[idx]

    test_ds = SimpleDataset(data_dir, "test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    class_names = test_ds.class_names
    n_classes   = len(class_names)
    print(f"  Test samples: {len(test_ds)}  |  Classes: {class_names}")

    # Load model
    print("[2/5] Loading model checkpoint...")
    ckpt_path = ckpt_dir / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        print("  Run 03_classify.py first.")
        return

    model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    # Inference
    print("[3/5] Running inference on test set...")
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  Evaluating"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_score = np.array(all_probs)

    # Metrics
    print("[4/5] Computing metrics & generating plots...")
    cm     = confusion_matrix(y_true, y_pred)
    kappa  = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                   target_names=class_names, output_dict=True)

    print(f"\n  Cohen's Kappa:      {kappa:.4f}")
    print(f"  Macro F1:           {report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted Accuracy:  {report['weighted avg']['f1-score']:.4f}")
    print("\n  Per-Class Report:")
    for cls in class_names:
        r = report[cls]
        print(f"    {cls:<20}  P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  F1={r['f1-score']:.3f}  "
              f"N={int(r['support'])}")

    # Specificity per class
    print("\n  Specificity per class:")
    for i, cls in enumerate(class_names):
        tn = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp + 1e-8)
        print(f"    {cls:<20}  {spec:.3f}")

    # Save report
    report_df = pd.DataFrame(report).T
    report_df["kappa"] = kappa
    report_df.to_csv(out_dir / "classification_report.csv")

    # Confusion matrix
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    print(f"  Confusion matrix saved.")

    # ROC curves
    plot_roc_curves(y_true, y_score, class_names, out_dir / "roc_curves.png")
    print(f"  ROC curves saved.")

    # Grad-CAM
    print("[5/5] Generating Grad-CAM visualisations...")
    try:
        cam_layer = get_gradcam_layer(model, model_name)
        gradcam   = GradCAM(model, cam_layer)
        plot_gradcam_grid(model, gradcam, test_ds, class_names,
                          device, n_gradcam, out_dir / "gradcam_grid.png")
    except Exception as e:
        print(f"  [WARN] Grad-CAM failed: {e}")

    # Summary JSON
    summary = {
        "model": model_name, "kappa": kappa,
        "macro_f1":    report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class":   {cls: report[cls] for cls in class_names},
    }
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete. All outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cataract classifier")
    parser.add_argument("--data_dir",   default="outputs/processed")
    parser.add_argument("--ckpt_dir",   default="outputs/classifier")
    parser.add_argument("--out_dir",    default="outputs/evaluation")
    parser.add_argument("--model",      default="efficientnet_b3")
    parser.add_argument("--n_gradcam",  type=int, default=4)
    args = parser.parse_args()
    main(args.data_dir, args.ckpt_dir, args.out_dir, args.model, args.n_gradcam)
