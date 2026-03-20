import subprocess
import sys
import os

# ── Install dependencies ───────────────────────────────────────────
subprocess.run("pip install -q -r requirements.txt", shell=True)
print("✅ Dependencies installed")

# ── Config ────────────────────────────────────────────────────────
DRIVE_OUT    = '/content/drive/MyDrive/EyeAI outputs'
DATA_DIR     = "data/dataset"
PROCESSED    = "outputs/processed"
CLASSIFIER   = "outputs/classifier"
EVALUATION   = "outputs/evaluation"
SEGMENTATION = "outputs/segmentation"
AUGMENTED    = "outputs/augmented"
LLM_OUT      = "outputs/llm_comparison"
MODELS       = ["efficientnet_b3", "resnet50", "vit_base_patch16_224"]

os.makedirs(DRIVE_OUT, exist_ok=True)

def run(cmd):
    print(f"\n{'='*60}\nRUNNING: {cmd}\n{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Failed: {cmd}")
        sys.exit(1)
    print("✅ Done")

def sync_to_drive():
    import shutil
    print("\n📂 Syncing outputs to Drive...")
    shutil.copytree('outputs', f'{DRIVE_OUT}/outputs', dirs_exist_ok=True)
    print(f"✅ Saved to {DRIVE_OUT}/outputs")

# ── Phase 1: EDA & Preprocessing ─────────────────────────────────
run(f"python 01_eda.py         --data_dir {DATA_DIR}  --out_dir outputs/eda")
run(f"python 02_preprocess.py  --data_dir {DATA_DIR}  --out_dir {PROCESSED}")

# ── Phase 2: Train & Evaluate all 3 models ───────────────────────
for model in MODELS:
    print(f"\n{'='*60}\n🧠 Training: {model}\n{'='*60}")
    run(f"python 03_classify.py --data_dir {PROCESSED} --out_dir {CLASSIFIER} --model {model} --epochs 30")
    run(f"python 04_evaluate.py --data_dir {PROCESSED} --ckpt_dir {CLASSIFIER} --out_dir {EVALUATION} --model {model}")

# ── Phase 3 & 4: Segmentation & GAN ──────────────────────────────
run(f"python 05_segment.py  --data_dir {PROCESSED} --out_dir {SEGMENTATION}")
run(f"python 06_augment.py  --data_dir {PROCESSED} --out_dir {AUGMENTED}")

# ── Sync checkpoints to Drive before LLM ─────────────────────────
sync_to_drive()

# ── Phase 6: LLM Benchmark (EfficientNet-B3 as CNN baseline) ─────
run(f"python 08_llm_classify_gemini.py \
    --data_dir   {PROCESSED} \
    --cnn_report {EVALUATION}/classification_report.csv \
    --out_dir    {LLM_OUT} \
    --mode       zero_shot \
    --max_images 5 \
    --model      gemini-2.5-flash \
    --sleep      15")

# ── Final sync ────────────────────────────────────────────────────
sync_to_drive()

print("\n🎉 All done! Check your Drive under 'EyeAI outputs/outputs/'")