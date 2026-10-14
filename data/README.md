# Data — not versioned

The raw data and the pre-trained models are **excluded from Git** (size and
licensing). This file explains how to recreate the `data/` folder locally.

## Expected layout

```
data/
├── Flipkart/
│   └── flipkart_com-ecommerce_sample_1050.csv   # text dataset (~1.7 MB)
├── Images/                                        # 1,050 product photos (.jpg)
└── use/
    └── universal-sentence-encoder_4/              # USE model (~960 MB, extracted)
```

## 1 · Flipkart dataset (text + images)

Source: **Flipkart Products** on Kaggle (public, CC0).
<https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products>

The OpenClassrooms project uses a 1,050-row sample
(`flipkart_com-ecommerce_sample_1050.csv`) and the matching product images.
Place the CSV in `data/Flipkart/` and the images in `data/Images/`.

```bash
# with the Kaggle CLI configured (~/.kaggle/kaggle.json)
kaggle datasets download -d PromptCloudHQ/flipkart-products -p data/Flipkart --unzip
```

## 2 · Universal Sentence Encoder (USE)

Downloaded on first use from TensorFlow Hub — no manual step needed:

```python
import tensorflow_hub as hub
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
```

To cache it under `data/use/` instead, download and extract the archive from the
[TF-Hub USE v4 page](https://tfhub.dev/google/universal-sentence-encoder/4).

## Why it's not in Git

- **Images** (~1,050 files) and the **USE model** (~960 MB) are large binaries,
  reproducible from their sources — versioning them bloats the repo for no gain.
- Keeping data out of Git is standard practice and keeps the history clean and
  the clone fast.
