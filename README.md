# EyeAI — Deep Learning for Early Cataract Detection

## Pre-requisites

1. Run via Google Colab (https://colab.research.google.com)
2. Set runtime to T4 GPU — `Runtime → Change runtime type → T4 GPU`
3. Add the following to Colab Secrets (key icon in left sidebar):
   - `GEMINI_API_KEY` from Google AI Studio (https://aistudio.google.com)
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` from Kaggle (https://www.kaggle.com)

---

## Execution

> **Note:** Cell 1 and Cell 2 must be run directly in a Google Colab cell. They cannot be moved into a `.py` script as they rely on Colab-specific UI features (`drive.mount`, `files.upload`).

**Cell 1 — Kaggle setup & download dataset**

```python
import os
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write('{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}')
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d jr2ngb/cataractdataset
!unzip -q cataractdataset.zip -d data/
```

**Cell 2 — Mount Drive & upload files**

```python
from google.colab import drive, files
drive.mount('/content/drive')
files.upload()  # upload all .py files + requirements.txt + run_pipeline.py
```

**Cell 3 — Run pipeline**

```python
from google.colab import userdata
import os
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')

!python run_pipeline.py
```

---

## Streamlit Deployment

Model checkpoints and class map are hosted on Hugging Face Hub and downloaded automatically at startup — no manual file transfers needed.

**Model repository:** https://huggingface.co/alcapps01/eyeai-models

### Steps

1. **No model download needed** — `06_app.py` fetches all `.pth` files and `class_map.json` from Hugging Face Hub automatically on first launch and caches them locally.

2. **Push code to GitHub** (no model files required in the repo)

```
EyeAI/
├── 06_app.py                          ← Streamlit app (root — required by Streamlit Cloud)
├── requirements.txt                   ← Dependencies (root — required by Streamlit Cloud)
├── packages.txt                       ← System dependencies (libgl1, libglib2.0-dev)
├── README.md
├── .devcontainer/
│   └── devcontainer.json              ← GitHub Codespaces config
└── scripts/
    ├── 01_eda.py
    ├── 02_preprocess.py
    ├── 03_classify.py
    ├── 04_evaluate.py
    ├── 05_llm_classify_gemini.py
    ├── 06_app.py
    └── run_pipeline.py
```

3. **Connect repo on [Streamlit Cloud](https://streamlit.io/cloud)**
4. **Set Python version to 3.11** in Streamlit Cloud dashboard settings
5. **Set main file path** to `06_app.py`
6. **Deploy** — on first boot the app downloads models from Hugging Face (~390MB), subsequent starts use the cache

---

## GitHub Codespaces

This repo includes a `.devcontainer` configuration. Click **Code → Open with Codespaces** on GitHub to launch a fully configured browser-based development environment with the app running automatically.

---

## Repository Notes

- Model `.pth` files and `class_map.json` are **not** stored in this repository — they are hosted at https://huggingface.co/alcapps01/eyeai-models
- If you retrain models via `run_pipeline.py`, upload the new `.pth` files and `class_map.json` to the Hugging Face repo to update the deployment
