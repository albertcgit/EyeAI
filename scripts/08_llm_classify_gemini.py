import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, cohen_kappa_score
)
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["normal", "cataract", "glaucoma", "retinal_disease"]
CLASS_LABELS = {c: i for i, c in enumerate(CLASS_NAMES)}
COLORS       = {"normal": "#5BAD6F", "cataract": "#E05A4E",
                "glaucoma": "#F4A642", "retinal_disease": "#9B59B6"}

SYSTEM_PROMPT = """You are an expert ophthalmologist specialising in retinal fundus image
interpretation. You will be shown one or more colour fundus photographs and must classify
each primary image into EXACTLY ONE of the following four categories:

  normal          — Healthy retina; no pathological features visible
  cataract        — Lens opacity causing image haze or reduced clarity of fundus structures
  glaucoma        — Optic disc cupping, cup-to-disc ratio > 0.6, or nerve fibre layer thinning
  retinal_disease — Other retinal pathology (e.g. diabetic retinopathy, macular degeneration)

Rules:
- Reply with ONLY a valid JSON object and nothing else.
- The JSON must have exactly two keys:
    "prediction": one of the four class names (exact lowercase, underscore for retinal_disease)
    "reasoning":  one sentence (≤ 20 words) explaining the key visual feature observed
- Do not include markdown, code fences, or any other text outside the JSON object.
"""

ZERO_SHOT_PROMPT = (
    "Classify this retinal fundus image into one of the four classes. "
    "Reply with the JSON object as instructed."
)

FEW_SHOT_PROMPT = (
    "The first four images are reference examples (one per class, labelled in the caption). "
    "Classify the FINAL image (labelled TARGET) using the same four-class scheme. "
    "Reply with the JSON object as instructed."
)


# ── Image utilities ────────────────────────────────────────────────────────────
def load_and_encode(path: Path, size: int = 512) -> str:
    """Read image, resize, and encode as base64 PNG for the API."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    h, w  = img.shape[:2]
    side  = min(h, w)
    img   = img[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
    img   = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def image_part(b64: str) -> dict:
    """Return a Gemini-compatible image part dict."""
    return {"mime_type": "image/png", "data": b64}


# ── API call ──────────────────────────────────────────────────────────────────
def call_gemini(model: genai.GenerativeModel, content: list,
                max_retries: int = 3) -> dict | None:
    """
    Sends a vision message to Gemini and parses the JSON response.
    Returns {"prediction": str, "reasoning": str} or None on failure.
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(content)
            raw = response.text.strip()

            # Strip any accidental markdown fences
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

            result = json.loads(raw)
            pred   = result.get("prediction", "").lower().replace(" ", "_")

            # Validate class name
            if pred in CLASS_NAMES:
                result["prediction"] = pred
                return result
            else:
                # Try fuzzy match
                for cls in CLASS_NAMES:
                    if cls.replace("_", "") in pred.replace("_", ""):
                        result["prediction"] = cls
                        return result
                print(f"  [WARN] Unrecognised class: '{pred}' — skipping")
                return None

        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error (attempt {attempt+1}): {e}")
            time.sleep(1.5)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                wait = 15 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] API call failed (attempt {attempt+1}): {e}")
                time.sleep(2)

    return None


# ── Dataset loader ─────────────────────────────────────────────────────────────
def load_test_paths(data_dir: Path, max_per_class: int) -> list[dict]:
    """Returns list of {path, label, class_idx} dicts from the test split."""
    records = []
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test split not found: {test_dir}")

    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        if label not in CLASS_LABELS:
            continue
        imgs = sorted([p for p in cls_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:max_per_class]
        for p in imgs:
            records.append({"path": p, "label": label,
                             "class_idx": CLASS_LABELS[label]})
    return records


def load_reference_images(data_dir: Path) -> dict[str, str]:
    """
    For few-shot: picks one image per class from the training split.
    Returns {class_name: base64_string}.
    """
    refs = {}
    train_dir = data_dir / "train"
    for cls in CLASS_NAMES:
        cls_dir = train_dir / cls
        if not cls_dir.exists():
            continue
        imgs = sorted([p for p in cls_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if imgs:
            refs[cls] = load_and_encode(imgs[0])
    return refs


# ── Evaluation modes ──────────────────────────────────────────────────────────
def run_zero_shot(model: genai.GenerativeModel, records: list[dict]) -> pd.DataFrame:
    """Zero-shot: one image per call, no examples."""
    results = []
    for rec in tqdm(records, desc="  Zero-shot"):
        b64     = load_and_encode(rec["path"])
        content = [image_part(b64), ZERO_SHOT_PROMPT]
        output  = call_gemini(model, content)
        if output:
            results.append({
                "path":      str(rec["path"]),
                "true":      rec["label"],
                "pred":      output["prediction"],
                "reasoning": output.get("reasoning", ""),
                "mode":      "zero_shot",
            })
        time.sleep(0.5)  # Rate-limit courtesy pause (free tier)
    return pd.DataFrame(results)


def run_few_shot(model: genai.GenerativeModel, records: list[dict],
                 refs: dict[str, str]) -> pd.DataFrame:
    """
    Few-shot: prepend one reference image per class, then classify target.
    """
    if not refs:
        print("  [WARN] No reference images available — skipping few-shot.")
        return pd.DataFrame()

    # Build the static reference block (reused for every query)
    ref_blocks = []
    for cls, b64 in refs.items():
        ref_blocks.append(image_part(b64))
        ref_blocks.append(f"[Reference — class: {cls}]")

    results = []
    for rec in tqdm(records, desc="  Few-shot "):
        b64     = load_and_encode(rec["path"])
        content = (ref_blocks
                   + [image_part(b64), "[TARGET image to classify]", FEW_SHOT_PROMPT])
        output  = call_gemini(model, content)
        if output:
            results.append({
                "path":      str(rec["path"]),
                "true":      rec["label"],
                "pred":      output["prediction"],
                "reasoning": output.get("reasoning", ""),
                "mode":      "few_shot",
            })
        time.sleep(0.6)
    return pd.DataFrame(results)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame, mode: str) -> dict:
    if df.empty:
        return {}
    y_true = [CLASS_LABELS[c] for c in df["true"]]
    y_pred = [CLASS_LABELS.get(c, -1) for c in df["pred"]]
    valid  = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
    if not valid:
        return {}
    yt, yp = zip(*valid)
    report = classification_report(yt, yp, target_names=CLASS_NAMES, output_dict=True)
    return {
        "mode":        mode,
        "accuracy":    round(accuracy_score(yt, yp), 4),
        "macro_f1":    round(f1_score(yt, yp, average="macro"), 4),
        "kappa":       round(cohen_kappa_score(yt, yp), 4),
        "per_class":   {cls: report.get(cls, {}) for cls in CLASS_NAMES},
        "n_evaluated": len(valid),
        "n_total":     len(df),
    }


# ── Comparison plot ────────────────────────────────────────────────────────────
def plot_comparison(metrics_list: list[dict], cnn_report_path: Path | None,
                    out_path: Path):
    """
    Bar chart comparing accuracy, macro-F1, and Kappa across:
      - CNN models (from 04_evaluate.py CSVs, if available)
      - LLM zero-shot
      - LLM few-shot
    """
    rows = []

    # Load CNN results if available
    if cnn_report_path and cnn_report_path.exists():
        cnn_df = pd.read_csv(cnn_report_path, index_col=0)
        if "f1-score" in cnn_df.columns:
            kappa_val = float(cnn_df.loc["normal", "kappa"]) \
                if "kappa" in cnn_df.columns else 0.0
            rows.append({
                "model":    "EfficientNet-B3\n(fine-tuned)",
                "accuracy": round(float(cnn_df.loc["weighted avg", "f1-score"]), 4),
                "macro_f1": round(float(cnn_df.loc["macro avg",    "f1-score"]), 4),
                "kappa":    kappa_val,
            })

    for m in metrics_list:
        if not m:
            continue
        label = "Gemini Vision\n(zero-shot)" if m["mode"] == "zero_shot" \
                else "Gemini Vision\n(few-shot)"
        rows.append({
            "model":    label,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "kappa":    m["kappa"],
        })

    if not rows:
        print("  [WARN] No data for comparison plot.")
        return

    plot_df  = pd.DataFrame(rows)
    metrics  = ["accuracy", "macro_f1", "kappa"]
    xlabels  = ["Accuracy", "Macro F1", "Cohen's Kappa"]
    n_models = len(plot_df)
    x        = np.arange(len(metrics))
    width    = 0.6 / n_models
    pal      = ["#2E9AB5", "#E05A4E", "#F4A642", "#5BAD6F", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(plot_df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width - (n_models - 1) * width / 2,
                      vals, width, label=row["model"],
                      color=pal[i % len(pal)], alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison: Fine-Tuned CNN vs. Vision LLM (Gemini)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(1.0, ls="--", color="grey", alpha=0.3, lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Comparison chart saved: {out_path}")


def plot_per_class_comparison(metrics_list: list[dict], out_path: Path):
    """Per-class F1 grouped bar chart for each LLM mode."""
    rows = []
    for m in metrics_list:
        if not m or not m.get("per_class"):
            continue
        label = "zero-shot" if m["mode"] == "zero_shot" else "few-shot"
        for cls in CLASS_NAMES:
            f1 = m["per_class"].get(cls, {}).get("f1-score", 0.0)
            rows.append({"mode": label, "class": cls, "f1": f1})

    if not rows:
        return

    df    = pd.DataFrame(rows)
    modes = df["mode"].unique()
    x     = np.arange(len(CLASS_NAMES))
    width = 0.35
    pal   = ["#2E9AB5", "#E05A4E"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (mode, col) in enumerate(zip(modes, pal)):
        vals = [df[(df.mode == mode) & (df["class"] == c)]["f1"].values[0]
                if len(df[(df.mode == mode) & (df["class"] == c)]) > 0 else 0
                for c in CLASS_NAMES]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=f"Gemini ({mode})", color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Class F1: Gemini Vision (Zero-shot vs Few-shot)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Per-class F1 chart saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_dir, cnn_report, out_dir, mode, max_images, model_name):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 6: Vision LLM Benchmark — Gemini ({model_name}) ===\n")
    print(f"  Mode: {mode}  |  Max images per class: {max_images}\n")

    # Init Gemini client (API key auto-read from GEMINI_API_KEY env var)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey\n"
            "Then run: export GEMINI_API_KEY=your_key_here"
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT
    )

    # Load test images
    print("[1/5] Loading test image paths...")
    records = load_test_paths(data_dir, max_per_class=max_images)
    print(f"  Loaded {len(records)} test images across {len(CLASS_NAMES)} classes")

    if not records:
        print("  [ERROR] No test images found. Run 02_preprocess.py first.")
        return

    # Load reference images for few-shot
    refs = {}
    if mode in ("few_shot", "both"):
        print("[2/5] Loading reference images for few-shot...")
        refs = load_reference_images(data_dir)
        print(f"  References: {list(refs.keys())}")

    # Run evaluations
    print("[3/5] Running Gemini Vision API evaluations...")
    all_results, all_metrics = [], []

    if mode in ("zero_shot", "both"):
        print("\n  --- Zero-Shot ---")
        zs_df = run_zero_shot(model, records)
        if not zs_df.empty:
            all_results.append(zs_df)
            m = compute_metrics(zs_df, "zero_shot")
            all_metrics.append(m)
            print(f"  Zero-shot  | Acc={m['accuracy']:.4f} | "
                  f"F1={m['macro_f1']:.4f} | Kappa={m['kappa']:.4f} "
                  f"| N={m['n_evaluated']}/{m['n_total']}")

    if mode in ("few_shot", "both"):
        print("\n  --- Few-Shot ---")
        fs_df = run_few_shot(model, records, refs)
        if not fs_df.empty:
            all_results.append(fs_df)
            m = compute_metrics(fs_df, "few_shot")
            all_metrics.append(m)
            print(f"  Few-shot   | Acc={m['accuracy']:.4f} | "
                  f"F1={m['macro_f1']:.4f} | Kappa={m['kappa']:.4f} "
                  f"| N={m['n_evaluated']}/{m['n_total']}")

    # Save raw predictions
    print("\n[4/5] Saving results...")
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(out_dir / "llm_predictions.csv", index=False)
        print(f"  Predictions saved: {out_dir / 'llm_predictions.csv'}")

        # Per-mode classification reports
        for m in all_metrics:
            if not m:
                continue
            mode_df = combined[combined["mode"] == m["mode"]]
            y_true  = [CLASS_LABELS[c] for c in mode_df["true"]]
            y_pred  = [CLASS_LABELS.get(c, -1) for c in mode_df["pred"]]
            valid   = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
            if valid:
                yt, yp = zip(*valid)
                rpt = classification_report(yt, yp, target_names=CLASS_NAMES)
                fn  = out_dir / f"report_{m['mode']}.txt"
                fn.write_text(rpt)
                print(f"  Report: {fn}")

    # Save metrics JSON
    with open(out_dir / "llm_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Plots
    print("[5/5] Generating comparison plots...")
    cnn_path = Path(cnn_report) if cnn_report else None
    plot_comparison(all_metrics, cnn_path, out_dir / "model_comparison.png")
    plot_per_class_comparison(all_metrics, out_dir / "per_class_f1.png")

    # Console summary table
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<35} {'Accuracy':>9} {'Macro F1':>9} {'Kappa':>8}")
    print("  " + "-" * 64)

    # CNN row (from saved report)
    cnn_path_obj = Path(cnn_report) if cnn_report else None
    if cnn_path_obj and cnn_path_obj.exists():
        try:
            cnn_df  = pd.read_csv(cnn_path_obj, index_col=0)
            cnn_acc = float(cnn_df.loc["weighted avg", "f1-score"])
            cnn_f1  = float(cnn_df.loc["macro avg",   "f1-score"])
            cnn_k   = float(cnn_df.loc["normal", "kappa"]) \
                      if "kappa" in cnn_df.columns else 0.0
            print(f"  {'EfficientNet-B3 (fine-tuned)':<35} {cnn_acc:>9.4f} "
                  f"{cnn_f1:>9.4f} {cnn_k:>8.4f}")
        except Exception:
            pass

    for m in all_metrics:
        if not m:
            continue
        label = f"Gemini ({m['mode'].replace('_', ' ')})"
        print(f"  {label:<35} {m['accuracy']:>9.4f} "
              f"{m['macro_f1']:>9.4f} {m['kappa']:>8.4f}")

    print("=" * 70)
    print(f"\nLLM comparison complete. Outputs saved to: {out_dir}\n")
    print("  Note: Set GEMINI_API_KEY environment variable before running.")
    print("  Free key available at: https://aistudio.google.com/app/apikey")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Google Gemini Vision API against fine-tuned CNN")
    parser.add_argument("--data_dir",   default="outputs/processed",
                        help="Preprocessed dataset directory")
    parser.add_argument("--cnn_report", default="outputs/evaluation/classification_report.csv",
                        help="CNN classification report CSV from 04_evaluate.py")
    parser.add_argument("--out_dir",    default="outputs/llm_comparison")
    parser.add_argument("--mode",       default="both",
                        choices=["zero_shot", "few_shot", "both"])
    parser.add_argument("--max_images", type=int, default=50,
                        help="Max images per class to evaluate (free tier limit awareness)")
    parser.add_argument("--model",      default="gemini-1.5-flash",
                        help="Gemini model name (gemini-1.5-flash is free tier)")
    args = parser.parse_args()
    main(args.data_dir, args.cnn_report, args.out_dir,
         args.mode, args.max_images, args.model)
