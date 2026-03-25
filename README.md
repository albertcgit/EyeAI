# EyeAI

## Pre-requisites

1. Run via Google Colab (https://colab.research.google.com)
2. Set runtime to T4 GPU — `Runtime → Change runtime type → T4 GPU`
3. Add the following to Colab Secrets (key icon in left sidebar):
   - `GEMINI_API_KEY` from Google AI Studio (https://aistudio.google.com)
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` from Kaggle (https://www.kaggle.com)

---

## Executing with Google Colab

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

1. Download model files from Drive to your local machine:
   - `outputs/classifier/efficientnet_b3_best.pth`
   - `outputs/classifier/resnet50_best.pth`
   - `outputs/classifier/vit_base_patch16_224_best.pth`
   - `outputs/classifier/class_map.json`

2. Push all files including model files to GitHub

```
EyeAI/
├── 06_app.py                              ← Streamlit app (root — required by Streamlit Cloud)
├── requirements.txt                       ← Dependencies (root — required by Streamlit Cloud)
├── packages.txt                           ← System dependencies (libgl1, libglib2.0-dev)
├── README.md
├── outputs/
│   └── classifier/
│       ├── efficientnet_b3_best.pth       ← Required for app
│       ├── resnet50_best.pth              ← Required for app
│       ├── vit_base_patch16_224_best.pth  ← Required for app
│       └── class_map.json                 ← Required for app
└── scripts/
    ├── 01_eda.py
    ├── 02_preprocess.py
    ├── 03_classify.py
    ├── 04_evaluate.py
    ├── 05_llm_classify_gemini.py
    ├── 06_app.py
    └── run_pipeline.py
```

3. Connect repo on [Streamlit Cloud](https://streamlit.io/cloud)
4. Set Python version to **3.11** in Streamlit Cloud dashboard settings
5. Set main file path to `06_app.py`
6. Deploy
