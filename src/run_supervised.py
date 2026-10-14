"""
P6 — Phase 2: supervised classification + multimodal fusion.

Turns the feasibility study into a proof: train a real classifier on
Word2Vec (text), VGG16 (image), and the FUSION of both, with a fixed seed and
a stratified train/test split. Reports accuracy + macro-F1 per modality.

Run with the dedicated env (numpy<2 + sklearn/gensim/tensorflow):
    /tmp/p6env/bin/python run_supervised.py

Outputs:
    docs/figures/supervised_accuracy.png
    docs/figures/fusion_confusion_matrix.png
    results_supervised.json   (metrics, for README/portfolio)
"""
import os, json, re, warnings
warnings.filterwarnings("ignore")
import numpy as np

SEED = 42
np.random.seed(SEED)

# Paths default to the repo layout. Override with env vars if your data lives
# elsewhere (e.g. on a fast local disk — recommended, since loading 1,050 images
# over a slow/removable drive can throw intermittent I/O errors).
DATA_CSV = os.environ.get("P6_CSV", "data/Flipkart/flipkart_com-ecommerce_sample_1050.csv")
IMG_DIR = os.environ.get("P6_IMAGES", "data/Images/")
OUT_DIR = os.environ.get("P6_OUT", "docs/figures")

import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

for pkg in ["stopwords", "punkt"]:
    nltk.download(pkg, quiet=True)

print("=" * 60)
print("1 · Load data & target")
df = pd.read_csv(DATA_CSV)
df["category"] = df["product_category_tree"].apply(lambda x: x[2:].split(">>")[0].strip())
cats = sorted(df["category"].unique())
cat2id = {c: i for i, c in enumerate(cats)}
y = df["category"].map(cat2id).values
print(f"   {len(df)} products · {len(cats)} categories")

# ── Text features: Word2Vec ──────────────────────────────────────────
print("2 · Text features — Word2Vec")
stop_words = set(stopwords.words("english"))
tok = RegexpTokenizer(r"\w+")
stem = PorterStemmer()

def clean(doc):
    doc = str(doc).lower().strip()
    t = [w for w in tok.tokenize(doc) if w not in stop_words and w.isalpha() and len(w) >= 3]
    return [stem.stem(w) for w in t]

df["tokens"] = df["description"].apply(clean)
w2v = Word2Vec(sentences=df["tokens"].tolist(), vector_size=300, window=5,
               min_count=2, workers=4, seed=SEED)

def docvec(tokens):
    v = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(v, axis=0) if v else np.zeros(300)

X_text = np.vstack(df["tokens"].apply(docvec).values)
print(f"   text features: {X_text.shape}")

# ── Image features: VGG16 penultimate layer ──────────────────────────
print("3 · Image features — VGG16 transfer learning (this takes a few minutes)")
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Model

base = VGG16(weights="imagenet")
feat_model = Model(inputs=base.inputs, outputs=base.layers[-2].output)  # 4096-d

import time

def img_feature(fname):
    path = os.path.join(IMG_DIR, fname)
    for attempt in range(5):           # retry on transient I/O errors
        try:
            img = load_img(path, target_size=(224, 224))
            x = preprocess_input(img_to_array(img)[None, ...])
            return feat_model.predict(x, verbose=0)[0]
        except OSError:
            time.sleep(1)
    raise OSError(f"could not read {path} after retries")

img_col = "image"
feats = []
for i, fname in enumerate(df[img_col].values):
    feats.append(img_feature(fname))
    if (i + 1) % 100 == 0:
        print(f"   {i+1}/{len(df)} images")
X_img_full = np.vstack(feats)
print(f"   raw image features: {X_img_full.shape}")

# PCA to tame 4096 dims (fit later only on train to avoid leakage) — handled in pipeline below

# ── Evaluation ───────────────────────────────────────────────────────
print("4 · Supervised evaluation (stratified 80/20, seed=42)")

def evaluate(X, name, use_pca=False):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    if use_pca:
        pca = PCA(n_components=200, random_state=SEED).fit(X_tr)
        X_tr, X_te = pca.transform(X_tr), pca.transform(X_te)
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    print(f"   {name:28} acc={acc:.3f}  macroF1={f1:.3f}")
    return acc, f1, (y_te, pred)

res = {}
res["text"] = evaluate(X_text, "Text (Word2Vec)")
res["image"] = evaluate(X_img_full, "Image (VGG16)", use_pca=True)

# Fusion: reduce image first (consistent), then concat with text
scaler_img = StandardScaler().fit(X_img_full)
X_img_pca = PCA(n_components=200, random_state=SEED).fit_transform(scaler_img.transform(X_img_full))
X_fusion = np.hstack([X_text, X_img_pca])
res["fusion"] = evaluate(X_fusion, "Fusion (text + image)")

# ── Save metrics ─────────────────────────────────────────────────────
metrics = {k: {"accuracy": round(v[0], 4), "macro_f1": round(v[1], 4)}
           for k, v in res.items()}
os.makedirs(OUT_DIR, exist_ok=True)
json.dump({"seed": SEED, "n": len(df), "categories": cats, "metrics": metrics},
          open("results_supervised.json", "w"), indent=2)
print("\nSaved results_supervised.json")

# ── Figures ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["Text\n(Word2Vec)", "Image\n(VGG16)", "Fusion\n(text+image)"]
accs = [res["text"][0], res["image"][0], res["fusion"][0]]
f1s = [res["text"][1], res["image"][1], res["fusion"][1]]
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#60a5fa")
b2 = ax.bar(x + w/2, f1s, w, label="Macro-F1", color="#a78bfa")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylim(0, 1); ax.set_ylabel("Score")
ax.set_title("Supervised classification — single modality vs multimodal fusion\n"
             "(7 categories, stratified 80/20, seed=42)", fontweight="bold", fontsize=12)
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "supervised_accuracy.png"), dpi=130, bbox_inches="tight")
plt.close()
print("wrote docs/figures/supervised_accuracy.png")

# Fusion confusion matrix
y_te, pred = res["fusion"][2]
cm = confusion_matrix(y_te, pred)
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(7)); ax.set_yticks(range(7))
short = [c[:18] for c in cats]
ax.set_xticklabels(short, rotation=45, ha="right"); ax.set_yticklabels(short)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Multimodal fusion — confusion matrix (test set)\n"
             f"accuracy {res['fusion'][0]:.0%}", fontweight="bold", fontsize=12)
for i in range(7):
    for j in range(7):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()*0.5 else "black", fontsize=9)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fusion_confusion_matrix.png"), dpi=130, bbox_inches="tight")
plt.close()
print("wrote docs/figures/fusion_confusion_matrix.png")
print("\nDONE.")
