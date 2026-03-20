import json
import sys
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "normal":          "#5BAD6F",
    "cataract":        "#E05A4E",
    "glaucoma":        "#F4A642",
    "retinal_disease": "#9B59B6",
}
RISK_ICONS = {
    "normal":          "✅",
    "cataract":        "⚠️",
    "glaucoma":        "🔶",
    "retinal_disease": "🔴",
}
CLINICAL_NOTES = {
    "normal":
        "No significant pathology detected. Routine follow-up recommended.",
    "cataract":
        "Lens opacity indicators present. Consider referral for slit-lamp examination "
        "and surgical consultation if vision is impaired.",
    "glaucoma":
        "Optic disc changes consistent with glaucoma risk. "
        "Recommend IOP measurement, visual field test, and specialist referral.",
    "retinal_disease":
        "Retinal pathology detected. Urgent referral to ophthalmologist recommended "
        "for detailed fundus examination and OCT imaging.",
}


# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier(ckpt_dir: str = "outputs/classifier",
                    model_name: str = "efficientnet_b3"):
    ckpt_path = Path(ckpt_dir) / f"{model_name}_best.pth"
    class_map_path = Path(ckpt_dir) / "class_map.json"

    if not ckpt_path.exists():
        return None, None, None

    with open(class_map_path) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    n_classes = len(idx_to_class)
    model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, idx_to_class, model_name


@st.cache_resource
def load_unet(seg_dir: str, target: str):
    ckpt_path = Path(seg_dir) / target / "unet_best.pth"
    if not ckpt_path.exists():
        return None

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
        def __init__(self, base_ch=32):
            super().__init__()
            self.enc1 = DoubleConv(3, base_ch); self.enc2 = DoubleConv(base_ch, base_ch*2)
            self.enc3 = DoubleConv(base_ch*2, base_ch*4); self.enc4 = DoubleConv(base_ch*4, base_ch*8)
            self.bottleneck = DoubleConv(base_ch*8, base_ch*16)
            self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
            self.dec4 = DoubleConv(base_ch*16, base_ch*8)
            self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
            self.dec3 = DoubleConv(base_ch*8, base_ch*4)
            self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
            self.dec2 = DoubleConv(base_ch*4, base_ch*2)
            self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
            self.dec1 = DoubleConv(base_ch*2, base_ch)
            self.head = nn.Conv2d(base_ch, 1, 1); self.pool = nn.MaxPool2d(2)
        def forward(self, x):
            e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2))
            e4=self.enc4(self.pool(e3)); b=self.bottleneck(self.pool(e4))
            d4=self.dec4(torch.cat([self.up4(b),e4],1)); d3=self.dec3(torch.cat([self.up3(d4),e3],1))
            d2=self.dec2(torch.cat([self.up2(d3),e2],1)); d1=self.dec1(torch.cat([self.up1(d2),e1],1))
            return torch.sigmoid(self.head(d1))

    unet = UNet(base_ch=32)
    unet.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    unet.eval()
    return unet


# ── Inference Helpers ─────────────────────────────────────────────────────────
def preprocess_for_inference(img_rgb: np.ndarray) -> torch.Tensor:
    tf = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return tf(image=img_rgb)["image"].unsqueeze(0)


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    img  = img_bgr[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
    img  = cv2.resize(img, (224, 224))
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def run_gradcam(model, img_tensor: torch.Tensor, target_class: int,
                model_name: str) -> np.ndarray:
    activations, gradients = {}, {}

    def save_act(m, inp, out):
        activations["val"] = out.detach()
    def save_grad(m, inp, out):
        gradients["val"] = out[0].detach()

    # Get target layer
    if "efficientnet" in model_name:
        layer = model.conv_head
    elif "resnet" in model_name:
        layer = list(model.layer4.children())[-1]
    else:
        for l in reversed(list(model.modules())):
            if isinstance(l, nn.Conv2d):
                layer = l; break

    h1 = layer.register_forward_hook(save_act)
    h2 = layer.register_full_backward_hook(save_grad)

    model.zero_grad()
    out = model(img_tensor)
    out[0, target_class].backward()
    h1.remove(); h2.remove()

    weights = gradients["val"].mean(dim=(2, 3), keepdim=True)
    cam     = (weights * activations["val"]).sum(dim=1).squeeze()
    cam     = torch.relu(cam).cpu().numpy()
    cam     = cv2.resize(cam, (224, 224))
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay_heatmap(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (img_rgb * (1 - alpha) + heatmap * alpha).clip(0, 255).astype(np.uint8)


def run_segmentation(unet, img_rgb: np.ndarray) -> np.ndarray:
    tf    = A.Compose([A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])
    t     = tf(image=img_rgb)["image"].unsqueeze(0)
    with torch.no_grad():
        mask = unet(t)[0, 0].numpy()
    return mask


def compute_cdr(mask: np.ndarray) -> float:
    disc_bin = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(disc_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    _, _, _, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return round(float(h * 0.45 / (h + 1e-8)), 3)


def mask_overlay(img_rgb: np.ndarray, mask: np.ndarray,
                 color: tuple = (0, 200, 100)) -> np.ndarray:
    overlay = img_rgb.copy()
    m = (mask > 0.5)
    overlay[m] = (overlay[m] * 0.5 + np.array(color) * 0.5).clip(0, 255)
    return overlay.astype(np.uint8)


# ── Streamlit App ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Cataract Screening AI",
        page_icon="👁️",
        layout="wide",
    )

    # Header
    st.markdown("""
        <h1 style='text-align:center; color:#1B3A6B;'>👁️ Cataract Screening AI</h1>
        <p style='text-align:center; color:#555; font-size:1.1em;'>
          Automated ocular disease detection from retinal fundus images<br>
          <em>Research prototype — not for clinical use</em>
        </p>
        <hr style='border:1px solid #E0E0E0; margin-bottom:24px;'>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    model_name = st.sidebar.selectbox(
        "Classifier Model",
        ["efficientnet_b3", "resnet50", "vit_base_patch16_224"],
    )
    ckpt_dir   = st.sidebar.text_input("Classifier checkpoint dir", "outputs/classifier")
    seg_dir    = st.sidebar.text_input("Segmentation model dir",    "outputs/segmentation")
    show_clahe = st.sidebar.checkbox("Show CLAHE-enhanced image",   value=True)
    conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Pipeline phases:**\n"
        "1. EDA: `01_eda.py`\n"
        "2. Preprocessing: `02_preprocess.py`\n"
        "3. Classifier: `03_classify.py`\n"
        "4. Evaluation: `04_evaluate.py`\n"
        "5. Segmentation: `05_segment.py`\n"
        "6. GAN augment: `06_augment.py`\n"
        "7. This app: `07_app.py`"
    )

    # Load models
    with st.spinner("Loading models..."):
        classifier, idx_to_class, loaded_model_name = load_classifier(ckpt_dir, model_name)
        disc_unet = load_unet(seg_dir, "disc")
        lens_unet = load_unet(seg_dir, "lens")

    if classifier is None:
        st.warning(
            "⚠️ No trained classifier found. "
            f"Expected: `{ckpt_dir}/{model_name}_best.pth`\n\n"
            "Run `python 03_classify.py` first, then relaunch this app."
        )
        st.stop()

    # Upload
    st.subheader("📁 Upload Fundus Image")
    uploaded = st.file_uploader(
        "Upload a retinal fundus photograph (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("Upload a fundus image above to begin screening.")
        # Added authors here even if nothing is uploaded so it stays at the bottom
        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#777;'>"
            "Developed by <b>Albert Levinson Canonza</b> & <b>Earl Tavera</b> | 2026"
            "</p>",
            unsafe_allow_html=True
        )
        return

    # Load and preprocess
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_bgr    = apply_clahe(img_bgr)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb    = cv2.resize(img_rgb, (224, 224))

    # ── Classification ──────────────────────────────────────────
    img_tensor = preprocess_for_inference(img_rgb)
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx   = int(probs.argmax())
    pred_class = idx_to_class.get(pred_idx, f"class_{pred_idx}")
    pred_conf  = float(probs[pred_idx])
    icon       = RISK_ICONS.get(pred_class, "🔬")
    color      = CLASS_COLORS.get(pred_class, "#888")

    # ── Grad-CAM ────────────────────────────────────────────────
    try:
        cam     = run_gradcam(classifier, img_tensor.requires_grad_(True),
                              pred_idx, loaded_model_name)
        cam_img = overlay_heatmap(img_rgb, cam)
        has_cam = True
    except Exception as e:
        has_cam = False

    # ── Segmentation & Biomarkers ────────────────────────────────
    disc_mask = run_segmentation(disc_unet, img_rgb) if disc_unet else None
    lens_mask = run_segmentation(lens_unet, img_rgb) if lens_unet else None
    cdr       = compute_cdr(disc_mask) if disc_mask is not None else None
    opacity   = float(lens_mask.mean()) if lens_mask is not None else None

    # ── Layout ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"{icon} Diagnosis: **{pred_class.replace('_', ' ').title()}**")

    # Confidence bar
    col_pred, col_note = st.columns([1, 2])
    with col_pred:
        st.markdown(f"""
            <div style='background:{color}22; border-left:5px solid {color};
                        padding:16px; border-radius:8px; margin-bottom:12px;'>
              <span style='font-size:1.4em; font-weight:bold; color:{color};'>
                {pred_conf * 100:.1f}% confidence
              </span>
            </div>
        """, unsafe_allow_html=True)

        if pred_conf < conf_threshold:
            st.warning(f"Low confidence ({pred_conf:.1%}). Result may be unreliable.")

    with col_note:
        st.info(f"🩺 **Clinical note:** {CLINICAL_NOTES.get(pred_class, '')}")

    # Probability breakdown
    st.markdown("**Probability breakdown:**")
    prob_cols = st.columns(len(idx_to_class))
    for i, (ci, cn) in enumerate(idx_to_class.items()):
        with prob_cols[i]:
            p = float(probs[ci]) * 100
            c = CLASS_COLORS.get(cn, "#888")
            st.markdown(
                f"<div style='text-align:center; padding:10px; "
                f"background:{c}22; border-radius:6px;'>"
                f"<b style='color:{c};'>{cn.replace('_',' ').title()}</b><br>"
                f"<span style='font-size:1.3em;'>{p:.1f}%</span></div>",
                unsafe_allow_html=True
            )

    # ── Image columns ────────────────────────────────────────────
    st.markdown("---")
    n_cols = sum([True, has_cam,
                  disc_mask is not None, lens_mask is not None])
    cols = st.columns(n_cols)
    col_idx = 0

    with cols[col_idx]:
        label = "CLAHE Enhanced" if show_clahe else "Input Image"
        st.image(img_rgb, caption=label, use_container_width=True)
    col_idx += 1

    if has_cam:
        with cols[col_idx]:
            st.image(cam_img, caption="Grad-CAM Attention", use_container_width=True)
        col_idx += 1

    if disc_mask is not None:
        disc_viz = mask_overlay(img_rgb, disc_mask, color=(0, 200, 100))
        with cols[col_idx]:
            st.image(disc_viz, caption="Optic Disc Segmentation", use_container_width=True)
        col_idx += 1

    if lens_mask is not None:
        lens_viz = mask_overlay(img_rgb, lens_mask, color=(200, 100, 0))
        with cols[col_idx]:
            st.image(lens_viz, caption="Lens Segmentation", use_container_width=True)

    # ── Biomarkers ───────────────────────────────────────────────
    if cdr is not None or opacity is not None:
        st.markdown("---")
        st.subheader("🔬 Extracted Biomarkers")
        bm_cols = st.columns(2)

        if cdr is not None:
            glaucoma_risk = "High" if cdr > 0.6 else ("Moderate" if cdr > 0.4 else "Low")
            risk_color    = {"High": "#E05A4E", "Moderate": "#F4A642", "Low": "#5BAD6F"}[glaucoma_risk]
            with bm_cols[0]:
                st.metric("Cup-to-Disc Ratio (CDR)", f"{cdr:.3f}")
                st.markdown(
                    f"Glaucoma risk: "
                    f"<span style='color:{risk_color}; font-weight:bold;'>{glaucoma_risk}</span>"
                    f" (CDR > 0.6 = high risk)",
                    unsafe_allow_html=True
                )

        if opacity is not None:
            opacity_pct  = opacity * 100
            cataract_sev = ("Severe" if opacity > 0.4 else
                            "Moderate" if opacity > 0.2 else "Mild")
            sev_color = {"Severe": "#E05A4E", "Moderate": "#F4A642", "Mild": "#5BAD6F"}[cataract_sev]
            with bm_cols[1]:
                st.metric("Lens Opacity Area", f"{opacity_pct:.1f}%")
                st.markdown(
                    f"Cataract severity: "
                    f"<span style='color:{sev_color}; font-weight:bold;'>{cataract_sev}</span>",
                    unsafe_allow_html=True
                )

    # ── Disclaimer ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='color:#999; font-size:0.85em; text-align:center;'>"
        "⚠️ <b>Research Prototype.</b> This tool is not a certified medical device and must not "
        "be used for clinical diagnosis. Always consult a qualified ophthalmologist."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Author Attribution ───────────────────────────────────────
    st.markdown(
        "<p style='text-align:center; color:#777; margin-top:30px;'>"
        "Developed by <b>Albert Levinson Canonza</b> & <b>Earl Tavera</b> | 2026"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
