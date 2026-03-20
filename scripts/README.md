EyeAI

Pre-requisites

1) Run via Google Colab ( https://colab.research.google.com )
2) Set runtime to T4 GPU — `Runtime → Change runtime type → T4 GPU`
3) Add the following to Colab Secrets (key icon in left sidebar):
	a) `GEMINI_API_KEY` from Google AI Studio ( https://aistudio.google.com )
	b) `KAGGLE_USERNAME` and `KAGGLE_KEY` — from Kaggle ( https://www.kaggle.com )

Execution

Note: Cell 1 and Cell 2 must be run directly in a Google Colab cell. They cannot be moved into a `.py` script as they rely on Colab-specific UI features (`drive.mount`, `files.upload`).


1) Run below script in Cell 1 — Kaggle setup & download dataset

import os
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write('{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}')
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d jr2ngb/cataractdataset
!unzip -q cataractdataset.zip -d data/
```

2) Run below script in Cell 2 — Mount Drive & upload files

from google.colab import drive, files
drive.mount('/content/drive')
files.upload()  # upload all .py files + requirements.txt + run_pipeline.py

3) Run below script in Cell 3 — Run pipeline

from google.colab import userdata
import os
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')

!python run_pipeline.py
Streamlit Deployment

1) Download model files from Drive to your local machine:
   a) `outputs/classifier/efficientnet_b3_best.pth`
   b) `outputs/classifier/class_map.json`
   c) `outputs/segmentation/disc/unet_best.pth`
   d) `outputs/segmentation/lens/unet_best.pth`
2) Push all files including model files to GitHub
3) Connect repo on [Streamlit Cloud](https://streamlit.io/cloud)
4) Set main file path to `07_app.py`
5) Deploy