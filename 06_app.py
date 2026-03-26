import json
import sys
import os
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
from huggingface_hub import hf_hub_download

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Normal":          "#5BAD6F",
    "Cataract":        "#E05A4E",
    "Glaucoma":        "#F4A642",
    "Retinal_disease": "#9B59B6",
}
RISK_ICONS = {
    "Normal":          "✅",
    "Cataract":        "⚠️",
    "Glaucoma":        "🔶",
    "Retinal_disease": "🔴",
}
CLINICAL_NOTES = {
    "Normal":
        "No significant pathology detected. Routine follow-up recommended.",
    "Cataract":
        "Lens opacity indicators present. Consider referral for slit-lamp examination "
        "and surgical consultation if vision is impaired.",
    "Glaucoma":
        "Optic disc changes consistent with glaucoma risk. "
        "Recommend IOP measurement, visual field test, and specialist referral.",
    "Retinal_disease":
        "Retinal pathology detected. Urgent referral to ophthalmologist recommended "
        "for detailed fundus examination and OCT imaging.",
}


# ── Model Loading ─────────────────────────────────────────────────────────────
HF_REPO = "alcapps01/eyeai-models"

MODEL_STATS = {
    "efficientnet_b3": {
        "display_name": "EfficientNet-B3",
        "accuracy": 64.84,
        "macro_f1": 0.593,
        "kappa": 0.473,
        "per_class_f1": {
            "Cataract": 0.727,
            "Glaucoma": 0.444,
            "Normal":   0.747,
            "Retinal Disease": 0.452,
        },
        "note": "Fast & efficient — good all-round baseline",
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "accuracy": 67.03,
        "macro_f1": 0.580,
        "kappa": 0.486,
        "per_class_f1": {
            "Cataract": 0.788,
            "Glaucoma": 0.483,
            "Normal":   0.776,
            "Retinal Disease": 0.273,
        },
        "note": "Best overall performer (highest kappa)",
    },
    "vit_base_patch16_224": {
        "display_name": "ViT-B/16",
        "accuracy": 61.54,
        "macro_f1": 0.530,
        "kappa": 0.407,
        "per_class_f1": {
            "Cataract": 0.788,
            "Glaucoma": 0.400,
            "Normal":   0.708,
            "Retinal Disease": 0.222,
        },
        "note": "Transformer-based — needs larger dataset to shine",
    },
}

@st.cache_resource
def load_classifier(ckpt_dir: str = "outputs/classifier",
                    model_name: str = "efficientnet_b3"):
    try:
        # Download from Hugging Face Hub (cached after first download)
        ckpt_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{model_name}_best.pth",
        )
        class_map_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="class_map.json",
        )
    except Exception as e:
        # Fallback to local path
        ckpt_path = Path(ckpt_dir) / f"{model_name}_best.pth"
        class_map_path = Path(ckpt_dir) / "class_map.json"
        if not Path(ckpt_path).exists():
            st.warning(f"Could not load model from Hugging Face or local path: {e}")
            return None, None, None

    with open(class_map_path) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    n_classes = len(idx_to_class)
    model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, idx_to_class, model_name


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
    ckpt_dir       = "outputs/classifier"  # fallback only
    show_clahe     = st.sidebar.checkbox("Show CLAHE-enhanced image", value=True)
    conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    stats = MODEL_STATS[model_name]
    st.sidebar.markdown(f"**{stats['display_name']}** — *{stats['note']}*")
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("Accuracy", f"{stats['accuracy']:.1f}%")
    col2.metric("Macro F1", f"{stats['macro_f1']:.3f}")
    col3.metric("Kappa", f"{stats['kappa']:.3f}")
    st.sidebar.markdown("**Per-class F1 (test set, n=91)**")
    for cls, f1 in stats["per_class_f1"].items():
        bar = int(f1 * 20)
        filled = "█" * bar
        empty  = "░" * (20 - bar)
        st.sidebar.markdown(
            f"`{cls:<16}` {filled}{empty} `{f1:.3f}`",
            unsafe_allow_html=False,
        )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Pipeline phases:**\n"
        "1. EDA: `01_eda.py`\n"
        "2. Preprocessing: `02_preprocess.py`\n"
        "3. Classifier: `03_classify.py`\n"
        "4. Evaluation: `04_evaluate.py`\n"
        "5. LLM Benchmark: `05_llm_classify_gemini.py`\n"
        "6. This app: `06_app.py`"
    )

    # Load models
    with st.spinner("Loading models..."):
        classifier, idx_to_class, loaded_model_name = load_classifier(ckpt_dir, model_name)

    if classifier is None:
        st.warning(
            "⚠️ No trained classifier found. "
            f"Could not load model `{model_name}` from Hugging Face Hub.\n\n"
            "Check that the model file exists at: https://huggingface.co/alcapps01/eyeai-models"
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
    cols = st.columns(2 if has_cam else 1)

    with cols[0]:
        label = "CLAHE Enhanced" if show_clahe else "Input Image"
        st.image(img_rgb, caption=label, use_container_width=True)

    if has_cam:
        with cols[1]:
            st.image(cam_img, caption="Grad-CAM Attention", use_container_width=True)

    # ── Disclaimer ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='color:#999; font-size:0.85em; text-align:center;'>"
        "⚠️ ⚠️ <b>Research Prototype.</b> This tool is not a certified medical device and must not ⚠️ ⚠️  "
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
