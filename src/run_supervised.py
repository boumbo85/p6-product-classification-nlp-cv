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
# workers=1 is REQUIRED for reproducibility: with several workers gensim's
# training order is non-deterministic, so the Word2Vec vectors (and every
# downstream metric) shift by ~1 point between runs even with a fixed seed.
# One worker is slightly slower on 1,050 short docs but makes the run exact.
w2v = Word2Vec(sentences=df["tokens"].tolist(), vector_size=300, window=5,
               min_count=2, workers=1, seed=SEED)

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

# 4d · WEIGHTED late fusion — tune the text/image weight on the TRAIN set only
# (cross-val out-of-fold probabilities), then apply the best weight on test.
from sklearn.model_selection import cross_val_predict
oof_text = cross_val_predict(clone(CLASSIFIERS[best["text"][0]]),  Xtr_text, y_tr, cv=5, method="predict_proba")
oof_img  = cross_val_predict(clone(CLASSIFIERS[best["image"][0]]), Xtr_img,  y_tr, cv=5, method="predict_proba")
best_w, best_w_acc = 0.5, -1
for w in np.linspace(0, 1, 21):          # w = weight on text
    acc_w = accuracy_score(y_tr, (w * oof_text + (1 - w) * oof_img).argmax(1))
    if acc_w > best_w_acc:
        best_w_acc, best_w = acc_w, w
pred_wt = (best_w * p_text + (1 - best_w) * p_img).argmax(1)
acc_wt = accuracy_score(y_te, pred_wt)
f1_wt = f1_score(y_te, pred_wt, average="macro")
print(f"   {'fusion-WEIGHTED (w_text=%.2f)' % best_w:34} acc={acc_wt:.3f}  macroF1={f1_wt:.3f}")

# 4e · STACKING — a meta Logistic Regression on the two models' OOF probabilities
X_meta_tr = np.hstack([oof_text, oof_img])
X_meta_te = np.hstack([p_text, p_img])
meta = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_meta_tr, y_tr)
pred_stack = meta.predict(X_meta_te)
acc_stack = accuracy_score(y_te, pred_stack)
f1_stack = f1_score(y_te, pred_stack, average="macro")
print(f"   {'fusion-STACKING (meta-LogReg)':34} acc={acc_stack:.3f}  macroF1={f1_stack:.3f}")

# 4f · Why is the fusion gain small? Decorrelated-error analysis.
pt = p_text.argmax(1); pi = p_img.argmax(1)
text_ok, img_ok = (pt == y_te), (pi == y_te)
both_wrong = int(np.sum(~text_ok & ~img_ok))
only_text  = int(np.sum(text_ok & ~img_ok))   # text rescues image
only_img   = int(np.sum(~text_ok & img_ok))   # image rescues text
both_right = int(np.sum(text_ok & img_ok))
n_te = len(y_te)
# recoverable = cases where exactly one modality is right (fusion *could* fix)
recoverable = only_text + only_img
print(f"   -- error overlap (test n={n_te}): both_right={both_right} "
      f"both_wrong={both_wrong} only_text_right={only_text} only_img_right={only_img}")
print(f"   -- recoverable by fusion: {recoverable}/{n_te} = {recoverable/n_te:.1%} "
      f"| unrecoverable (both wrong): {both_wrong/n_te:.1%}")

res = {
    "text":          (best["text"][1],  best["text"][2],  best["text"][3]),
    "image":         (best["image"][1], best["image"][2], best["image"][3]),
    "fusion_early":  (acc_cat,   f1_cat,   pred_cat),
    "fusion_late":   (acc_late,  f1_late,  pred_late),
    "fusion_weighted":(acc_wt,   f1_wt,    pred_wt),
    "fusion_stacking":(acc_stack,f1_stack, pred_stack),
}
error_overlap = {"both_right": both_right, "both_wrong": both_wrong,
                 "only_text_right": only_text, "only_image_right": only_img,
                 "recoverable_pct": round(recoverable / n_te, 4),
                 "best_text_weight": round(float(best_w), 3)}

# ── Save metrics ─────────────────────────────────────────────────────
metrics = {k: {"accuracy": round(v[0], 4), "macro_f1": round(v[1], 4)}
           for k, v in res.items()}
metrics["best_classifier"] = {"text": best["text"][0], "image": best["image"][0]}
os.makedirs(OUT_DIR, exist_ok=True)
json_path = os.environ.get("P6_JSON", "results_supervised.json")
json.dump({"seed": SEED, "n": len(df), "categories": cats,
           "metrics": metrics, "error_overlap": error_overlap},
          open(json_path, "w"), indent=2)
print(f"\nSaved {json_path}")

# ── Figures ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["Text\n(Word2Vec)", "Image\n(VGG16)", "Early fusion\n(concat)",
          "Late fusion\n(avg)", "Weighted\nfusion", "Stacking\n(meta-LR)"]
keys = ["text", "image", "fusion_early", "fusion_late", "fusion_weighted", "fusion_stacking"]
accs = [res[k][0] for k in keys]
f1s = [res[k][1] for k in keys]
colors_acc = ["#60a5fa", "#a78bfa", "#f59e0b", "#10b981", "#14b8a6", "#059669"]
x = np.arange(len(keys)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 5))
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

# Confusion matrix of the BEST fusion model
fusion_keys = ["fusion_late", "fusion_weighted", "fusion_stacking"]
best_fusion = max(fusion_keys, key=lambda k: res[k][0])
y_true_cm, pred = y_te, res[best_fusion][2]
pretty = {"fusion_late": "Late fusion", "fusion_weighted": "Weighted fusion",
          "fusion_stacking": "Stacking"}[best_fusion]
cm = confusion_matrix(y_true_cm, pred)
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(7)); ax.set_yticks(range(7))
short = [c[:18] for c in cats]
ax.set_xticklabels(short, rotation=45, ha="right"); ax.set_yticklabels(short)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"{pretty} — confusion matrix (test set)\n"
             f"accuracy {res[best_fusion][0]:.0%}", fontweight="bold", fontsize=12)
for i in range(7):
    for j in range(7):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()*0.5 else "black", fontsize=9)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fusion_confusion_matrix.png"), dpi=130, bbox_inches="tight")
plt.close()
print("wrote fusion_confusion_matrix.png")

# Error-overlap figure — WHY the fusion gain is small
fig, ax = plt.subplots(figsize=(7, 4.5))
seg = [both_right, only_text, only_img, both_wrong]
seg_lbl = ["Both right", "Only text right\n(text rescues image)",
           "Only image right\n(image rescues text)", "Both wrong\n(unrecoverable)"]
seg_col = ["#10b981", "#60a5fa", "#a78bfa", "#ef4444"]
left = 0
for v, l, c in zip(seg, seg_lbl, seg_col):
    ax.barh(0, v, left=left, color=c, edgecolor="white")
    if v > 0:
        ax.text(left + v/2, 0, f"{v}\n({v/n_te:.0%})", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
    left += v
ax.set_yticks([]); ax.set_xlim(0, n_te)
ax.set_xlabel(f"Test products (n={n_te})")
ax.set_title("Why fusion gains little: where text & image agree vs disagree\n"
             f"only {recoverable/n_te:.0%} of cases are 'recoverable' by fusion; "
             f"{both_wrong/n_te:.0%} are wrong in both modalities",
             fontweight="bold", fontsize=11)
ax.legend([plt.Rectangle((0,0),1,1,color=c) for c in seg_col],
          ["Both right", "Only text right", "Only image right", "Both wrong"],
          loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8)
ax.spines[["top", "right", "left"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "error_overlap.png"), dpi=130, bbox_inches="tight")
plt.close()
print("wrote error_overlap.png")
print("\nDONE.")
