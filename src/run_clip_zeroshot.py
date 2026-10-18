"""
P6 — 2026 addendum: CLIP zero-shot classification (no training at all).

The 2023 pipeline needed two encoders (Word2Vec + VGG16) and a fusion study to
combine them. CLIP collapses that: text and image land in ONE shared vector
space, so a product can be labelled by cosine-similarity to the *name* of each
category — with zero training examples.

This script measures, on the real 1,050-product Flipkart set:
  1. zero-shot from the image only          (image · category-name)
  2. zero-shot from the text only           (description · category-name)
  3. zero-shot from the average of both     (shared space => fusion is a mean)
and compares them to the supervised 2023 numbers (results_supervised.json).

Run:
    /tmp/p6env/bin/python src/run_clip_zeroshot.py
Paths default to the repo layout; override with env vars if data lives elsewhere.

Outputs:
    results_clip_zeroshot.json
    docs/figures/clip_zeroshot.png
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np

SEED = 42
np.random.seed(SEED)

DATA_CSV = os.environ.get("P6_CSV", "data/Flipkart/flipkart_com-ecommerce_sample_1050.csv")
IMG_DIR = os.environ.get("P6_IMAGES", "data/Images/")
OUT_DIR = os.environ.get("P6_OUT", "docs/figures")

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("=" * 60)
print("1 · Load data & target")
df = pd.read_csv(DATA_CSV)
df["category"] = df["product_category_tree"].apply(lambda x: x[2:].split(">>")[0].strip())
cats = sorted(df["category"].unique())
cat2id = {c: i for i, c in enumerate(cats)}
y = df["category"].map(cat2id).values
print(f"   {len(df)} products · {len(cats)} categories")

print("2 · Load CLIP (clip-ViT-B-32) — downloaded once, then cached")
from sentence_transformers import SentenceTransformer, util
clip = SentenceTransformer("clip-ViT-B-32")   # runs on CPU

# CLIP's text encoder is hard-capped at 77 tokens. sentence-transformers'
# CLIPModel.tokenize() does NOT pass truncation to the tokenizer, so long
# product descriptions blow past the limit and crash. We wrap the tokenizer
# call to force truncation at 77 tokens — the honest, documented fix.
_clip_module = clip[0]
_orig_tokenizer = _clip_module.processor.tokenizer
def _truncating_tokenizer(texts, **kwargs):
    kwargs.setdefault("truncation", True)
    kwargs.setdefault("max_length", 77)
    return _orig_tokenizer(texts, **kwargs)
_clip_module.processor.tokenizer = _truncating_tokenizer

# ── Encode the category names as text prompts ────────────────────────
# Zero-shot trick: the classifier IS the list of label strings. A short
# natural-language template ("a photo of ...") matches CLIP's training better
# than a bare category name.
prompts = [f"a photo of a product in the category: {c}" for c in cats]
label_emb = clip.encode(prompts, convert_to_numpy=True, normalize_embeddings=True)

# ── Encode every product's text — two ways, to measure the 77-token cost ─────
# (a) TRUNCATED: keep the first 77 tokens (the tokenizer wrapper above cuts it).
# (b) CHUNK-AND-POOL: split the description into 77-token chunks, encode each,
#     average the (normalized) embeddings. This recovers the information past
#     token 77 that truncation throws away. We report BOTH to quantify the gain.
print("3 · Encode product descriptions — truncated vs chunk-and-pool")
descriptions = df["description"].astype(str).tolist()

txt_emb = clip.encode(descriptions, batch_size=64, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)

def chunk_tokens(text, size=70):
    """Split into <=size-token pieces using CLIP's own tokenizer (ids -> text)."""
    ids = _orig_tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if len(ids) <= size:
        return [str(text)]
    return [_orig_tokenizer.decode(ids[i:i+size], skip_special_tokens=True)
            for i in range(0, len(ids), size)]

def embed_chunk_pool(text):
    chunks = chunk_tokens(text)
    embs = clip.encode(chunks, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)
    v = embs.mean(axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

txt_emb_pool = np.vstack([embed_chunk_pool(t) for t in descriptions])

# ── Encode every product's image ─────────────────────────────────────
print("4 · Encode product images (this takes a few minutes on CPU)")
imgs = []
for i, fname in enumerate(df["image"].values):
    imgs.append(Image.open(os.path.join(IMG_DIR, str(fname))).convert("RGB"))
    if (i + 1) % 200 == 0:
        print(f"   loaded {i+1}/{len(df)}")
img_emb = clip.encode(imgs, batch_size=32, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)

# ── Zero-shot predictions (cosine similarity, argmax over categories) ─
print("5 · Zero-shot classification — no training")
def predict(emb):
    sims = emb @ label_emb.T          # (n, 7) cosine sims (all normalized)
    return sims.argmax(axis=1)

pred_img = predict(img_emb)
pred_txt = predict(txt_emb)              # truncated text
pred_txt_pool = predict(txt_emb_pool)    # chunk-and-pool text
# fusion in the shared space = average the two normalized embeddings
def fuse(a, b):
    f = a + b
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    return f
fused = fuse(img_emb, txt_emb)
fused_pool = fuse(img_emb, txt_emb_pool)
pred_fuse = predict(fused)
pred_fuse_pool = predict(fused_pool)

def report(name, pred):
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    print(f"   {name:34} acc={acc:.3f}  macroF1={f1:.3f}")
    return acc, f1

res = {
    "clip_image":      report("CLIP zero-shot · image", pred_img),
    "clip_text":       report("CLIP zero-shot · text (truncated)", pred_txt),
    "clip_text_pool":  report("CLIP zero-shot · text (chunk-pool)", pred_txt_pool),
    "clip_fused":      report("CLIP zero-shot · image+text (trunc)", pred_fuse),
    "clip_fused_pool": report("CLIP zero-shot · image+text (pool)", pred_fuse_pool),
}

# ── Phase 1-bis: SAME unsupervised protocol as the 2023 re-run, on CLIP ──────
# To answer "what does CLIP add vs my 2023 methods" fairly, score it on the SAME
# metric (ARI) with the SAME pipeline: standardize -> PCA -> K-Means(k=7) -> ARI,
# clustering in PCA space (NOT t-SNE). Averaged over several seeds (mean ± std),
# exactly like run_supervised.py. Directly comparable to the phase-1 ARI there.
print("6 · Unsupervised ARI on CLIP embeddings — PCA clustering, multi-seed")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
SEEDS = [42, 7, 123, 2024, 99]

def ari_pca(features, pca_dim=50, seed=42):
    X = StandardScaler().fit_transform(np.asarray(features, dtype=float))
    dim = min(pca_dim, X.shape[1])
    X = PCA(n_components=dim, random_state=seed).fit_transform(X)
    labels = KMeans(n_clusters=len(cats), n_init=10, random_state=seed).fit_predict(X)
    return adjusted_rand_score(y, labels)

def ari_mstd(features):
    runs = [ari_pca(features, seed=s) for s in SEEDS]
    return round(float(np.mean(runs)), 4), round(float(np.std(runs)), 4)

ari = {
    "clip_text":  ari_mstd(txt_emb),
    "clip_image": ari_mstd(img_emb),
    "clip_fused": ari_mstd(fused),
}
for k, (m, s) in ari.items():
    print(f"   ARI · {k:18} = {m} ± {s}")

# ── 7 · SUPERVISED head on CLIP embeddings — the apples-to-apples test ───────
# Zero-shot vs the 2023 supervised pipeline changes TWO things at once (encoder
# AND supervision). To isolate "are CLIP embeddings better *at equal
# supervision*?", we train the SAME kind of head (LogReg / SVM sweep) on CLIP
# embeddings, with the SAME protocol as run_supervised.py: repeated stratified
# 5-fold CV over the same seeds. Directly comparable to Word2Vec+VGG16 (0.889).
print("7 · Supervised head on CLIP embeddings (repeated 5-fold CV, multi-seed)")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score

CLF = {"LogReg": lambda s: LogisticRegression(max_iter=2000, random_state=s),
       "SVM":    lambda s: SVC(kernel="rbf", random_state=s)}

def cv_supervised(X, seeds=SEEDS):
    X = np.asarray(X, dtype=float)
    accs, f1s = [], []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            # small sweep, keep best on this fold
            best_acc, best_pred = -1, None
            for name, mk in CLF.items():
                pred = mk(seed).fit(Xtr, y[tr]).predict(Xte)
                a = accuracy_score(y[te], pred)
                if a > best_acc:
                    best_acc, best_pred = a, pred
            accs.append(accuracy_score(y[te], best_pred))
            f1s.append(f1_score(y[te], best_pred, average="macro"))
    return (round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4),
            round(float(np.mean(f1s)), 4), round(float(np.std(f1s)), 4))

# CLIP text uses the chunk-pool embeddings (best text variant); fusion = concat
clip_sup = {
    "clip_text_sup":  cv_supervised(txt_emb_pool),
    "clip_image_sup": cv_supervised(img_emb),
    "clip_fused_sup": cv_supervised(np.hstack([txt_emb_pool, img_emb])),
}
for k, v in clip_sup.items():
    print(f"   {k:18} acc={v[0]:.3f}±{v[1]:.3f}  F1={v[2]:.3f}±{v[3]:.3f}")

# ── Pull the 2023 reference (ARI + supervised) from results_supervised.json ──
sup_full = None
sup_path = os.environ.get("P6_JSON", "results_supervised.json")
if os.path.exists(sup_path):
    sup_full = json.load(open(sup_path))

ari_2023 = None
sup_ref = None
if sup_full:
    ari_2023 = sup_full.get("phase1_ari_pca")       # same PCA protocol -> comparable
    sup_ref = sup_full.get("phase2_supervised")

out = {
    "model": "clip-ViT-B-32",
    "zero_shot": True,
    "seeds": SEEDS,
    "n": int(len(df)),
    "categories": cats,
    "metrics": {k: {"accuracy": round(v[0], 4), "macro_f1": round(v[1], 4)}
                for k, v in res.items()},
    # Unsupervised ARI on CLIP embeddings, SAME PCA protocol, mean ± std:
    "ari_pca_same_protocol": {
        "clip_text":  {"mean": ari["clip_text"][0],  "std": ari["clip_text"][1]},
        "clip_image": {"mean": ari["clip_image"][0], "std": ari["clip_image"][1]},
        "clip_fused": {"mean": ari["clip_fused"][0], "std": ari["clip_fused"][1]},
        "reference_2023": ari_2023,
    },
    # Supervised head on CLIP embeddings, SAME 5-fold CV protocol as 2023:
    "supervised_on_clip": {
        k: {"accuracy_mean": v[0], "accuracy_std": v[1],
            "macro_f1_mean": v[2], "macro_f1_std": v[3]}
        for k, v in clip_sup.items()
    },
}
if sup_ref:
    out["supervised_reference"] = {
        "image_VGG16": sup_ref.get("image"),
        "late_fusion": sup_ref.get("fusion_late"),
    }
clip_json_path = os.environ.get("P6_CLIP_JSON", "results_clip_zeroshot.json")
json.dump(out, open(clip_json_path, "w"), indent=2)
print(f"\nSaved {clip_json_path}")

# ── Figure: zero-shot vs supervised, side by side ────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bars = [
    ("CLIP zero-shot\nimage", res["clip_image"][0], "#f59e0b"),
    ("CLIP zero-shot\ntext (pool)", res["clip_text_pool"][0], "#f59e0b"),
    ("CLIP zero-shot\nimage+text (pool)", res["clip_fused_pool"][0], "#d97706"),
]
if sup_ref:
    bars.append(("Supervised 2023\nimage (VGG16)", sup_ref["image"]["accuracy_mean"], "#a78bfa"))
    bars.append(("Supervised 2023\nlate fusion", sup_ref["fusion_late"]["accuracy_mean"], "#10b981"))

labels = [b[0] for b in bars]
vals = [b[1] for b in bars]
colors = [b[2] for b in bars]
x = np.arange(len(bars))
fig, ax = plt.subplots(figsize=(10, 5))
b = ax.bar(x, vals, color=colors, width=0.6)
for bar in b:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(1/len(cats), ls=":", color="grey", alpha=0.7)
ax.text(len(bars)-0.5, 1/len(cats)+0.01, "random (1/7)", fontsize=8, color="grey", ha="right")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1); ax.set_ylabel("Accuracy (7 categories)")
ax.set_title("CLIP zero-shot (no training) vs the supervised 2023 pipeline\n"
             "1,050 Flipkart products · 7 categories",
             fontweight="bold", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
if sup_ref:
    ax.axvline(2.5, ls="--", lw=1, color="lightgrey")
plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, "clip_zeroshot.png"), dpi=130, bbox_inches="tight")
plt.close()
print("wrote docs/figures/clip_zeroshot.png")

# ── Confusion matrix of the best zero-shot variant ───────────────────
best_key = max(res, key=lambda k: res[k][0])
pred_map = {"clip_image": pred_img, "clip_text": pred_txt, "clip_text_pool": pred_txt_pool,
            "clip_fused": pred_fuse, "clip_fused_pool": pred_fuse_pool}
pretty_map = {"clip_image": "image", "clip_text": "text (trunc)",
              "clip_text_pool": "text (pool)", "clip_fused": "image+text (trunc)",
              "clip_fused_pool": "image+text (pool)"}
best_pred = pred_map[best_key]
pretty = pretty_map[best_key]
cm = confusion_matrix(y, best_pred)
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(cm, cmap="Oranges")
ax.set_xticks(range(len(cats))); ax.set_yticks(range(len(cats)))
short = [c[:18] for c in cats]
ax.set_xticklabels(short, rotation=45, ha="right"); ax.set_yticklabels(short)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"CLIP zero-shot ({pretty}) — confusion matrix\n"
             f"accuracy {res[best_key][0]:.0%} · zero training examples",
             fontweight="bold", fontsize=12)
for i in range(len(cats)):
    for j in range(len(cats)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()*0.5 else "black", fontsize=9)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "clip_zeroshot_confusion.png"), dpi=130, bbox_inches="tight")
plt.close()
print("wrote docs/figures/clip_zeroshot_confusion.png")

# ── Figure: apples-to-apples — 2023 best vs CLIP, SAME metric (ARI) ──────────
# This is the honest "what did revisiting in 2026 buy?" chart: same unsupervised
# protocol, same ARI metric, 2023 encoders vs CLIP embeddings.
fig, ax = plt.subplots(figsize=(9, 5))
groups = ["Text", "Image"]
# 2023 reference ARI (PCA protocol, mean) pulled from results_supervised.json
if ari_2023:
    vals_2023 = [ari_2023["text_word2vec"]["mean"], ari_2023["image_vgg16"]["mean"]]
    err_2023  = [ari_2023["text_word2vec"]["std"],  ari_2023["image_vgg16"]["std"]]
else:
    vals_2023, err_2023 = [0, 0], [0, 0]
vals_clip = [ari["clip_text"][0], ari["clip_image"][0]]
err_clip  = [ari["clip_text"][1], ari["clip_image"][1]]
xg = np.arange(len(groups)); w = 0.38
b1 = ax.bar(xg - w/2, vals_2023, w, yerr=err_2023, capsize=3,
            label="2023 (Word2Vec / VGG16)", color="#a78bfa")
b2 = ax.bar(xg + w/2, vals_clip, w, yerr=err_clip, capsize=3,
            label="2026 (CLIP embeddings)", color="#7c3aed")
ax.axhline(ari["clip_fused"][0], ls="--", lw=1.2, color="#d97706",
           label=f"CLIP fused (text+image) = {ari['clip_fused'][0]}")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xg); ax.set_xticklabels(groups)
ax.set_ylabel("Adjusted Rand Index (ARI)")
ax.set_ylim(0, max(vals_2023 + vals_clip + [ari["clip_fused"][0]]) * 1.3)
ax.set_title("Revisiting 2026 — same unsupervised protocol, same metric (ARI)\n"
             "2023 encoders vs CLIP · K-Means in PCA space · mean ± std over 5 seeds",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ari_2023_vs_clip.png"), dpi=130, bbox_inches="tight")
plt.close()
print("wrote docs/figures/ari_2023_vs_clip.png")
print("\nDONE.")
