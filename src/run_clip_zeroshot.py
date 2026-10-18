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

# ── Encode every product's text ──────────────────────────────────────
# Descriptions are truncated to 77 tokens by the tokenizer wrapper above.
# The first 77 tokens hold the product's defining terms; losing the tail is a
# real limitation of CLIP on verbose product text, reported honestly.
print("3 · Encode product descriptions (truncated to CLIP's 77-token limit)")
descriptions = df["description"].astype(str).tolist()
txt_emb = clip.encode(descriptions,
                      batch_size=64, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)

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
pred_txt = predict(txt_emb)
# fusion in the shared space = average the two normalized embeddings
fused = img_emb + txt_emb
fused /= np.linalg.norm(fused, axis=1, keepdims=True)
pred_fuse = predict(fused)

def report(name, pred):
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    print(f"   {name:28} acc={acc:.3f}  macroF1={f1:.3f}")
    return acc, f1

res = {
    "clip_image": report("CLIP zero-shot · image", pred_img),
    "clip_text":  report("CLIP zero-shot · text", pred_txt),
    "clip_fused": report("CLIP zero-shot · image+text", pred_fuse),
}

# ── Compare to the supervised 2023 numbers, if available ─────────────
sup = None
sup_path = os.environ.get("P6_JSON", "results_supervised.json")
if os.path.exists(sup_path):
    sup = json.load(open(sup_path))["metrics"]

out = {
    "model": "clip-ViT-B-32",
    "zero_shot": True,
    "seed": SEED,
    "n": int(len(df)),
    "categories": cats,
    "metrics": {k: {"accuracy": round(v[0], 4), "macro_f1": round(v[1], 4)}
                for k, v in res.items()},
}
if sup:
    out["supervised_reference"] = {
        "image_VGG16_SVM": sup.get("image"),
        "late_fusion": sup.get("fusion_late"),
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
    ("CLIP zero-shot\ntext", res["clip_text"][0], "#f59e0b"),
    ("CLIP zero-shot\nimage+text", res["clip_fused"][0], "#d97706"),
]
if sup:
    bars.append(("Supervised 2023\nVGG16+SVM", sup["image"]["accuracy"], "#a78bfa"))
    bars.append(("Supervised 2023\nlate fusion", sup["fusion_late"]["accuracy"], "#10b981"))

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
# visually separate the two families
if sup:
    ax.axvline(2.5, ls="--", lw=1, color="lightgrey")
plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, "clip_zeroshot.png"), dpi=130, bbox_inches="tight")
plt.close()
print("wrote docs/figures/clip_zeroshot.png")

# ── Confusion matrix of the best zero-shot variant ───────────────────
best_key = max(res, key=lambda k: res[k][0])
best_pred = {"clip_image": pred_img, "clip_text": pred_txt, "clip_fused": pred_fuse}[best_key]
pretty = {"clip_image": "image", "clip_text": "text", "clip_fused": "image+text"}[best_key]
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
print("\nDONE.")
