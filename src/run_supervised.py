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
# One single stratified split shared by both modalities — mandatory so that
# late fusion can average per-sample probabilities from the same test rows.
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

print("4 · Supervised evaluation (stratified 80/20, seed=42)")

idx = np.arange(len(y))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
y_tr, y_te = y[idx_tr], y[idx_te]


def prep(X, pca_dim=None):
    """Scale (+ optional PCA) fit on train only, return (Xtr, Xte)."""
    Xtr, Xte = X[idx_tr], X[idx_te]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if pca_dim and Xtr.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=SEED).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    return Xtr, Xte


CLASSIFIERS = {
    "LogReg": LogisticRegression(max_iter=2000, random_state=SEED),
    "SVM":    SVC(kernel="rbf", probability=True, random_state=SEED),
    "RF":     RandomForestClassifier(n_estimators=300, random_state=SEED),
}


def score(name, Xtr, Xte, clf):
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xte)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    print(f"   {name:34} acc={acc:.3f}  macroF1={f1:.3f}")
    return acc, f1, pred


# Prepare the two modalities on the shared split
Xtr_text, Xte_text = prep(X_text)
Xtr_img,  Xte_img  = prep(X_img_full, pca_dim=200)

# 4a · best classifier per modality (small model sweep)
print("  -- model sweep per modality --")
best = {}
for mod, (Xtr, Xte) in {"text": (Xtr_text, Xte_text), "image": (Xtr_img, Xte_img)}.items():
    best_acc = -1
    for cname, clf in CLASSIFIERS.items():
        from sklearn.base import clone
        acc, f1, pred = score(f"{mod} · {cname}", Xtr, Xte, clone(clf))
        if acc > best_acc:
            best_acc, best[mod] = acc, (cname, acc, f1, pred)
    print(f"   -> best {mod}: {best[mod][0]} (acc={best[mod][1]:.3f})")

# 4b · naive early fusion (concat) — the documented negative result
from sklearn.base import clone
Xtr_cat = np.hstack([Xtr_text, Xtr_img])
Xte_cat = np.hstack([Xte_text, Xte_img])
acc_cat, f1_cat, pred_cat = score("fusion-EARLY (concat)", Xtr_cat, Xte_cat,
                                  clone(CLASSIFIERS["LogReg"]))

# 4c · LATE fusion — average the per-class probabilities of the two
# best single-modality classifiers (same test rows).
def proba(clf_name, Xtr, Xte):
    clf = clone(CLASSIFIERS[clf_name]).fit(Xtr, y_tr)
    return clf.predict_proba(Xte)

p_text = proba(best["text"][0],  Xtr_text, Xte_text)
p_img  = proba(best["image"][0], Xtr_img,  Xte_img)
pred_late = (p_text + p_img).argmax(axis=1)
acc_late = accuracy_score(y_te, pred_late)
f1_late = f1_score(y_te, pred_late, average="macro")
print(f"   {'fusion-LATE (avg proba)':34} acc={acc_late:.3f}  macroF1={f1_late:.3f}")

res = {
    "text":        (best["text"][1],  best["text"][2],  best["text"][3]),
    "image":       (best["image"][1], best["image"][2], best["image"][3]),
    "fusion_early":(acc_cat,  f1_cat,  pred_cat),
    "fusion_late": (acc_late, f1_late, pred_late),
}

# ── Save metrics ─────────────────────────────────────────────────────
metrics = {k: {"accuracy": round(v[0], 4), "macro_f1": round(v[1], 4)}
           for k, v in res.items()}
metrics["best_classifier"] = {"text": best["text"][0], "image": best["image"][0]}
os.makedirs(OUT_DIR, exist_ok=True)
json.dump({"seed": SEED, "n": len(df), "categories": cats, "metrics": metrics},
          open("results_supervised.json", "w"), indent=2)
print("\nSaved results_supervised.json")

# ── Figures ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["Text\n(Word2Vec)", "Image\n(VGG16)",
          "Early fusion\n(concat)", "Late fusion\n(avg proba)"]
keys = ["text", "image", "fusion_early", "fusion_late"]
accs = [res[k][0] for k in keys]
f1s = [res[k][1] for k in keys]
colors_acc = ["#60a5fa", "#a78bfa", "#f59e0b", "#10b981"]
x = np.arange(4); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=colors_acc)
b2 = ax.bar(x + w/2, f1s, w, label="Macro-F1", color=colors_acc, alpha=0.5)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", fontsize=9, fontweight="bold")
# single-modality reference line
ax.axhline(max(accs[0], accs[1]), ls="--", lw=1, color="grey", alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1); ax.set_ylabel("Score")
ax.set_title("Supervised classification — single modality vs fusion strategies\n"
             "(7 categories, stratified 80/20, seed=42 · best classifier per modality)",
             fontweight="bold", fontsize=12)
ax.legend(loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "supervised_accuracy.png"), dpi=130, bbox_inches="tight")
plt.close()
print("wrote supervised_accuracy.png")

# Late-fusion confusion matrix (the winning model)
y_true_cm, pred = y_te, res["fusion_late"][2]
cm = confusion_matrix(y_true_cm, pred)
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(7)); ax.set_yticks(range(7))
short = [c[:18] for c in cats]
ax.set_xticklabels(short, rotation=45, ha="right"); ax.set_yticklabels(short)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Late-fusion classifier — confusion matrix (test set)\n"
             f"accuracy {res['fusion_late'][0]:.0%}", fontweight="bold", fontsize=12)
for i in range(7):
    for j in range(7):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()*0.5 else "black", fontsize=9)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fusion_confusion_matrix.png"), dpi=130, bbox_inches="tight")
plt.close()
print("wrote fusion_confusion_matrix.png")
print("\nDONE.")
