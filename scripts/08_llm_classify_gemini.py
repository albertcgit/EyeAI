import argparse
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, cohen_kappa_score
)
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["normal", "cataract", "glaucoma", "retinal_disease"]
CLASS_LABELS = {c: i for i, c in enumerate(CLASS_NAMES)}

SYSTEM_PROMPT = (
    "You are an expert ophthalmologist classifying retinal fundus images. "
    "Classify into EXACTLY ONE of: normal, cataract, glaucoma, retinal_disease. "
    "Reply ONLY with a JSON object with two keys: "
    "'prediction' (exact class name) and "
    "'reasoning' (one sentence, max 20 words). "
    "No markdown, no code fences, no extra text."
)

ZERO_SHOT_PROMPT = (
    "Classify this retinal fundus image. "
    "Reply with the JSON object as instructed."
)

FEW_SHOT_PROMPT = (
    "The first four images are labelled reference examples (one per class). "
    "Classify the FINAL image labelled TARGET. "
    "Reply with the JSON object as instructed."
)

IMG_SIZE = 256   # 256px keeps payload small without losing diagnostic detail


# Image utilities
def load_image_bytes(path: Path, size: int = IMG_SIZE) -> bytes:
    """Read image, centre-crop to square, resize, return PNG bytes."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    h, w   = img.shape[:2]
    s      = min(h, w)
    img    = img[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
    img    = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def image_part(img_bytes: bytes) -> types.Part:
    return types.Part.from_bytes(data=img_bytes, mime_type="image/png")


def text_part(text: str) -> types.Part:
    return types.Part.from_text(text=text)


# API call
def call_gemini(client: genai.Client, model_name: str,
                parts: list, max_retries: int = 3) -> dict | None:
    """
    Sends a vision message to Gemini and parses the JSON response.
    Returns {"prediction": str, "reasoning": str} or None on failure.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=300,   # must be >= 300 to avoid truncated JSON
                    temperature=0.1,
                )
            )
            raw = re.sub(r"```(?:json)?", "", response.text.strip()).strip("`").strip()

            result = json.loads(raw)
            pred   = result.get("prediction", "").lower().replace(" ", "_")

            if pred in CLASS_NAMES:
                result["prediction"] = pred
                return result

            # Fuzzy match fallback
            for cls in CLASS_NAMES:
                if cls.replace("_", "") in pred.replace("_", ""):
                    result["prediction"] = cls
                    return result

            print(f"  [WARN] Unrecognised class: '{pred}' — skipping")
            return None

        except json.JSONDecodeError:
            print(f"  [WARN] JSON parse error (attempt {attempt+1})")
            time.sleep(5)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                wait = 30 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
            elif "503" in err_str or "unavailable" in err_str.lower():
                wait = 10 * (attempt + 1)
                print(f"  [UNAVAILABLE] Model busy. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {e}")
                time.sleep(5)

    return None


# Dataset loader
def load_test_paths(data_dir: Path, max_per_class: int) -> list[dict]:
    """Returns list of {path, label, class_idx} dicts from the test split."""
    records  = []
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

    total = len(records)
    print(f"  Total API calls needed: {total} (zero_shot) or {total * 2} (both modes)")
    return records


def load_reference_images(data_dir: Path) -> dict[str, bytes]:
    """For few-shot: picks one image per class from the training split."""
    refs      = {}
    train_dir = data_dir / "train"
    for cls in CLASS_NAMES:
        cls_dir = train_dir / cls
        if not cls_dir.exists():
            continue
        imgs = sorted([p for p in cls_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if imgs:
            refs[cls] = load_image_bytes(imgs[0])
    return refs


# Evaluation modes
def run_zero_shot(client: genai.Client, model_name: str,
                  records: list[dict], sleep_sec: int) -> pd.DataFrame:
    """Zero-shot: one image per call, no examples."""
    results = []
    for rec in tqdm(records, desc="  Zero-shot"):
        parts  = [image_part(load_image_bytes(rec["path"])),
                  text_part(ZERO_SHOT_PROMPT)]
        output = call_gemini(client, model_name, parts)
        if output:
            results.append({
                "path":      str(rec["path"]),
                "true":      rec["label"],
                "pred":      output["prediction"],
                "reasoning": output.get("reasoning", ""),
                "mode":      "zero_shot",
            })
            print(f"  {rec['path'].name} → {output['prediction']}")
        else:
            print(f"  [SKIP] {rec['path'].name}")
        time.sleep(sleep_sec)
    return pd.DataFrame(results)


def run_few_shot(client: genai.Client, model_name: str,
                 records: list[dict], refs: dict[str, bytes],
                 sleep_sec: int) -> pd.DataFrame:
    """Few-shot: prepend one reference image per class, then classify target."""
    if not refs:
        print("  [WARN] No reference images — skipping few-shot.")
        return pd.DataFrame()

    ref_parts = []
    for cls, img_bytes in refs.items():
        ref_parts.append(image_part(img_bytes))
        ref_parts.append(text_part(f"[Reference — class: {cls}]"))

    results = []
    for rec in tqdm(records, desc="  Few-shot "):
        parts  = ref_parts + [
            image_part(load_image_bytes(rec["path"])),
            text_part("[TARGET image to classify]"),
            text_part(FEW_SHOT_PROMPT)
        ]
        output = call_gemini(client, model_name, parts)
        if output:
            results.append({
                "path":      str(rec["path"]),
                "true":      rec["label"],
                "pred":      output["prediction"],
                "reasoning": output.get("reasoning", ""),
                "mode":      "few_shot",
            })
            print(f"  {rec['path'].name} → {output['prediction']}")
        else:
            print(f"  [SKIP] {rec['path'].name}")
        time.sleep(sleep_sec)
    return pd.DataFrame(results)


# Metrics
def compute_metrics(df: pd.DataFrame, mode: str) -> dict:
    if df.empty:
        return {}
    y_true = [CLASS_LABELS[c] for c in df["true"]]
    y_pred = [CLASS_LABELS.get(c, -1) for c in df["pred"]]
    valid  = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
    if not valid:
        return {}
    yt, yp = zip(*valid)
    report = classification_report(yt, yp, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    return {
        "mode":        mode,
        "accuracy":    round(accuracy_score(yt, yp), 4),
        "macro_f1":    round(f1_score(yt, yp, average="macro", zero_division=0), 4),
        "kappa":       round(cohen_kappa_score(yt, yp), 4),
        "per_class":   {cls: report.get(cls, {}) for cls in CLASS_NAMES},
        "n_evaluated": len(valid),
        "n_total":     len(df),
    }


# Plots
def plot_comparison(metrics_list: list[dict], cnn_report_path: Path | None,
                    out_path: Path):
    """Bar chart: Accuracy / Macro F1 / Kappa across CNN and Gemini models."""
    rows = []

    if cnn_report_path and cnn_report_path.exists():
        summary_path = cnn_report_path.parent / "evaluation_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path) as f:
                summary = json.load(f)
            rows.append({
                "model":    "EfficientNet-B3\n(fine-tuned)",
                "accuracy": round(float(summary.get("weighted_f1", 0.0)), 4),
                "macro_f1": round(float(summary.get("macro_f1", 0.0)), 4),
                "kappa":    round(float(summary.get("kappa", 0.0)), 4),
            })
        elif cnn_report_path.exists():
            cnn_df = pd.read_csv(cnn_report_path, index_col=0)
            if "f1-score" in cnn_df.columns:
                acc_row = "accuracy" if "accuracy" in cnn_df.index else "weighted avg"
                rows.append({
                    "model":    "EfficientNet-B3\n(fine-tuned)",
                    "accuracy": round(float(cnn_df.loc[acc_row, "f1-score"]) if acc_row != "accuracy"
                                      else float(cnn_df.loc["accuracy"].iloc[0]), 4),
                    "macro_f1": round(float(cnn_df.loc["macro avg", "f1-score"]), 4),
                    "kappa":    0.0,
                })

    for m in metrics_list:
        if not m:
            continue
        label = "Gemini Vision\n(zero-shot)" if m["mode"] == "zero_shot" \
                else "Gemini Vision\n(few-shot)"
        rows.append({"model": label, "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"], "kappa": m["kappa"]})

    if not rows:
        print("  [WARN] No data for comparison plot.")
        return

    plot_df  = pd.DataFrame(rows)
    metrics  = ["accuracy", "macro_f1", "kappa"]
    xlabels  = ["Accuracy", "Macro F1", "Cohen's Kappa"]
    n_models = len(plot_df)
    x        = np.arange(len(metrics))
    width    = 0.6 / n_models
    pal      = ["#2E9AB5", "#E05A4E", "#F4A642"]

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
    ax.set_ylim(-0.15, 1.12)
    ax.set_title("Model Comparison: EfficientNet-B3 (Fine-tuned) vs Gemini Vision (Zero-shot)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Score")
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Comparison chart saved: {out_path}")


def plot_per_class_comparison(metrics_list: list[dict],
                               cnn_report_path: Path | None, out_path: Path):
    """Per-class F1 grouped bar chart: CNN vs Gemini."""
    rows = []

    if cnn_report_path and cnn_report_path.exists():
        summary_path = cnn_report_path.parent / "evaluation_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path) as f:
                summary = json.load(f)
            per_class = summary.get("per_class", {})
            for cls in CLASS_NAMES:
                if cls in per_class:
                    f1 = per_class[cls].get("f1-score", 0.0)
                    rows.append({"mode": "EfficientNet-B3", "class": cls,
                                 "f1": round(float(f1), 4)})
        else:
            cnn_df = pd.read_csv(cnn_report_path, index_col=0)
            for cls in CLASS_NAMES:
                if cls in cnn_df.index:
                    rows.append({"mode": "EfficientNet-B3", "class": cls,
                                 "f1": round(float(cnn_df.loc[cls, "f1-score"]), 4)})

    for m in metrics_list:
        if not m or not m.get("per_class"):
            continue
        label = "Gemini (zero-shot)" if m["mode"] == "zero_shot" else "Gemini (few-shot)"
        for cls in CLASS_NAMES:
            f1 = m["per_class"].get(cls, {}).get("f1-score", 0.0)
            rows.append({"mode": label, "class": cls, "f1": f1})

    if not rows:
        return

    df    = pd.DataFrame(rows)
    modes = df["mode"].unique()
    x     = np.arange(len(CLASS_NAMES))
    width = 0.25
    pal   = ["#2E9AB5", "#E05A4E", "#F4A642"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (mode, col) in enumerate(zip(modes, pal)):
        vals = [df[(df.mode == mode) & (df["class"] == c)]["f1"].values[0]
                if len(df[(df.mode == mode) & (df["class"] == c)]) > 0 else 0
                for c in CLASS_NAMES]
        offset = (i - (len(modes) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=mode, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_title("Per-Class F1: EfficientNet-B3 vs Gemini Vision",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Per-class F1 chart saved: {out_path}")


# Main
def main(data_dir, cnn_report, out_dir, mode, max_images, model_name, sleep_sec):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 6: Vision LLM Benchmark — Gemini ({model_name}) ===\n")
    print(f"  Mode: {mode}  |  Max images per class: {max_images}")
    est_calls = max_images * 4 * (2 if mode == "both" else 1)
    est_time  = round(est_calls * sleep_sec / 60, 1)
    print(f"  Estimated API calls: {est_calls}  |  Estimated time: ~{est_time} mins\n")

    # Init client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey\n"
            "In Colab, add to Secrets then run:\n"
            "  import os; from google.colab import userdata\n"
            "  os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')"
        )
    client = genai.Client(api_key=api_key)

    # Verify model is available
    try:
        available = [m.name for m in client.models.list()]
        full_name = f"models/{model_name}"
        if full_name not in available:
            print(f"  [WARN] {model_name} not found. Available models:")
            for m in available:
                if "gemini" in m and "flash" in m:
                    print(f"    {m}")
            print("  Update --model argument and retry.")
            return
        print(f"  Model confirmed available: {full_name}\n")
    except Exception as e:
        print(f"  [WARN] Could not verify model list: {e}")

    # Load data
    print("[1/5] Loading test image paths...")
    records = load_test_paths(data_dir, max_per_class=max_images)
    print(f"  Loaded {len(records)} test images across {len(CLASS_NAMES)} classes")

    if not records:
        print("  [ERROR] No test images found. Run 02_preprocess.py first.")
        return

    refs = {}
    if mode in ("few_shot", "both"):
        print("[2/5] Loading reference images for few-shot...")
        refs = load_reference_images(data_dir)
        print(f"  References loaded: {list(refs.keys())}")
    else:
        print("[2/5] Skipping reference images (zero_shot mode)")

    # Run evaluations
    print("[3/5] Running Gemini Vision API evaluations...")
    all_results, all_metrics = [], []

    if mode in ("zero_shot", "both"):
        print("\n  --- Zero-Shot ---")
        zs_df = run_zero_shot(client, model_name, records, sleep_sec)
        if not zs_df.empty:
            all_results.append(zs_df)
            m = compute_metrics(zs_df, "zero_shot")
            all_metrics.append(m)
            print(f"\n  Zero-shot  | Acc={m['accuracy']:.4f} | "
                  f"F1={m['macro_f1']:.4f} | Kappa={m['kappa']:.4f} "
                  f"| N={m['n_evaluated']}/{m['n_total']}")

    if mode in ("few_shot", "both"):
        print("\n  --- Few-Shot ---")
        fs_df = run_few_shot(client, model_name, records, refs, sleep_sec)
        if not fs_df.empty:
            all_results.append(fs_df)
            m = compute_metrics(fs_df, "few_shot")
            all_metrics.append(m)
            print(f"\n  Few-shot   | Acc={m['accuracy']:.4f} | "
                  f"F1={m['macro_f1']:.4f} | Kappa={m['kappa']:.4f} "
                  f"| N={m['n_evaluated']}/{m['n_total']}")

    # Save results
    print("\n[4/5] Saving results...")
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(out_dir / "llm_predictions.csv", index=False)
        print(f"  Predictions saved: {out_dir / 'llm_predictions.csv'}")

        for m in all_metrics:
            if not m:
                continue
            mode_df = combined[combined["mode"] == m["mode"]]
            y_true  = [CLASS_LABELS[c] for c in mode_df["true"]]
            y_pred  = [CLASS_LABELS.get(c, -1) for c in mode_df["pred"]]
            valid   = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
            if valid:
                yt, yp = zip(*valid)
                rpt = classification_report(yt, yp, target_names=CLASS_NAMES,
                                            zero_division=0)
                fn  = out_dir / f"report_{m['mode']}.txt"
                fn.write_text(rpt)
                print(f"  Classification report: {fn}")

    with open(out_dir / "llm_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrics JSON saved.")

    # Plots
    print("\n[5/5] Generating comparison plots...")
    cnn_path = Path(cnn_report) if cnn_report else None
    plot_comparison(all_metrics, cnn_path, out_dir / "model_comparison.png")
    plot_per_class_comparison(all_metrics, cnn_path, out_dir / "per_class_f1.png")

    # Summary table
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<35} {'Accuracy':>9} {'Macro F1':>9} {'Kappa':>8}")
    print("  " + "-" * 64)

    cnn_path_obj = Path(cnn_report) if cnn_report else None
    if cnn_path_obj and cnn_path_obj.exists():
        try:
            import json
            summary_path = cnn_path_obj.parent / "evaluation_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                cnn_acc = float(summary.get("weighted_f1", 0.0))
                cnn_f1  = float(summary.get("macro_f1", 0.0))
                cnn_k   = float(summary.get("kappa", 0.0))
            else:
                cnn_df  = pd.read_csv(cnn_path_obj, index_col=0)
                cnn_acc = float(cnn_df.loc["weighted avg", "f1-score"])
                cnn_f1  = float(cnn_df.loc["macro avg",   "f1-score"])
                cnn_k   = 0.0
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Google Gemini Vision API against fine-tuned CNN")
    parser.add_argument("--data_dir",   default="outputs/processed")
    parser.add_argument("--cnn_report", default="outputs/evaluation/classification_report.csv")
    parser.add_argument("--out_dir",    default="outputs/llm_comparison")
    parser.add_argument("--mode",       default="zero_shot",
                        choices=["zero_shot", "few_shot", "both"],
                        help="zero_shot uses fewest API calls (recommended for free tier)")
    parser.add_argument("--max_images", type=int, default=5,
                        help="Images per class — 5 = 20 total calls")
    parser.add_argument("--model",      default="gemini-2.5-flash",
                        help="Gemini model name (confirmed working: gemini-2.5-flash)")
    parser.add_argument("--sleep",      type=int, default=15,
                        help="Seconds to wait between API calls (15 recommended for free tier)")
    args = parser.parse_args()
    main(args.data_dir, args.cnn_report, args.out_dir,
         args.mode, args.max_images, args.model, args.sleep)