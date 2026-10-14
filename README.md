# Multimodal Product Classification — NLP + Computer Vision

> Can a product's **category** be recovered automatically from its text description and its photo — with no manual labelling?
> A feasibility study comparing **6 text** and **2 image** feature-extraction methods on 1,050 Flipkart products.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-VGG16-ff6f00)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![status](https://img.shields.io/badge/status-completed-success)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

*OpenClassrooms Data Scientist — Project 6 · 2023*

🇫🇷 [Version française](#-version-française) · 🇬🇧 [English below](#-english)

---

## 🇬🇧 English

### The question

An e-commerce marketplace lists products that sellers categorise by hand —
slow, inconsistent, error-prone. **Could categories be assigned automatically**
from what we already have: a text description and a photo? This project answers
that *before* building a production classifier, by measuring how much category
signal each data source actually carries.

### Approach — an honest, unsupervised benchmark

Rather than jump to a supervised model, I treated it as a **feasibility study**.
The same protocol is applied to every feature set, so the numbers are directly
comparable:

```
features → PCA (denoise) → t-SNE (2D) → K-Means (k=7) → Adjusted Rand Index
```

The **Adjusted Rand Index (ARI)** scores how well the 7 unsupervised clusters
match the 7 true categories (0 = random, 1 = perfect). No labels are used to
build the features — only to score them.

### Results

| Modality | Method | ARI |
|---|---|:--:|
| Text | CountVectorizer (bag-of-words) | 0.333 |
| Text | TF-IDF | 0.240 |
| **Text** | **Word2Vec** | **0.475** 🏆 |
| Text | BERT (`bert-base-uncased`, [CLS]) | 0.182 |
| Text | BERT (TF-Hub) | 0.289 |
| Text | USE (Universal Sentence Encoder) | 0.423 |
| Image | SIFT (bag-of-visual-words) | 0.053 |
| **Image** | **CNN — VGG16 transfer learning** | **0.460** 🏆 |

![ARI benchmark across the 8 methods](docs/figures/ari_benchmark.png)

The best image model, VGG16, is worth a closer look. Clustering alone gives
ARI 0.460; once the clusters are optimally matched to the true categories
(Hungarian assignment), that is a **66% accuracy** — on 7 classes, from images
only, with no supervision. The confusion matrix shows the model confuses exactly
the categories a human would: *Home Furnishing* leaks into *Baby Care* and
*Home Decor* (visually similar textiles and decor), while *Watches* is near-perfect
(138/150).

![VGG16 confusion matrix](docs/figures/cnn_confusion_matrix.png)

### What the numbers say

- **Feasibility: yes.** The best text (0.475) and image (0.460) methods both
  recover categories well above chance — with zero manual labelling.
- **Simpler beat heavier on this data.** Word2Vec outscored BERT (0.18–0.29).
  Product descriptions are keyword bags, not sentences, so BERT's contextual
  strength goes unused when frozen. *Match the model to the data, not to the hype.*
- **Transfer learning is the unlock for images.** VGG16 (0.460) vs SIFT (0.053) —
  a ~9× gain. Learned features transfer across domains; hand-crafted ones don't.
- **The two modalities are complementary** (text ≈ image), which is the case for
  fusing them — tested in phase 2 below.

### Phase 2 — supervised classification & fusion

Phase 1 answers *"is the signal there?"*. Phase 2 answers *"how well can we
actually classify?"* — a supervised classifier (Logistic Regression), fixed
seed, stratified 80/20 split, accuracy + macro-F1.

| Features | Accuracy | Macro-F1 |
|---|:--:|:--:|
| **Image — VGG16** | **0.824** | 0.821 |
| **Text — Word2Vec** | **0.814** | 0.809 |
| Fusion (text ⊕ image, naïve concat) | 0.781 | 0.776 |

![Supervised accuracy — single modality vs fusion](docs/figures/supervised_accuracy.png)

Two results worth stating plainly:

- **~0.82 accuracy on 7 classes**, from a single modality — feasibility is now a
  hard number, not a hunch.
- **Naïve fusion makes it *worse* (0.78).** Concatenating 300-dim text and
  200-dim image features hands the classifier 500 features of unequal scale and
  density; the weak dimensions add noise faster than signal. *Fusion is a
  modelling decision, not an `np.hstack`.* The right next step is late fusion
  (combine per-modality probabilities) or a learned joint space — see
  [notebook 02](notebooks/02_supervised_fusion.ipynb).

> **Reproducibility note.** The original 2023 study reported slightly different
> ARI values between the slides and the notebook (e.g. USE ≈ 0.49 vs 0.42)
> because random seeds were not fixed. Everything here is **re-run with
> `seed=42` fixed across Word2Vec, PCA, t-SNE, K-Means and the train/test
> split**, so the numbers are stable and reproducible. The phase-1 ARI table
> reflects the notebook values.

### Tech stack

**NLP** — NLTK (tokenise, stop-words, Porter stemming, WordNet lemmatisation),
scikit-learn (CountVectorizer, TF-IDF), gensim (Word2Vec), Hugging Face
Transformers (BERT), TensorFlow Hub (USE).
**Computer Vision** — OpenCV (SIFT), TensorFlow / Keras (VGG16 transfer learning).
**ML & viz** — scikit-learn (PCA, t-SNE, K-Means, ARI), pandas, matplotlib, seaborn.

### Repository structure

```
p6-product-classification-nlp-cv/
├── README.md                                   # this file
├── requirements.txt                            # pinned dependencies
├── LICENSE                                      # MIT
├── results_supervised.json                     # phase-2 metrics (reproducible)
├── notebooks/
│   ├── 01_product_classification_text_image.ipynb   # phase 1 — unsupervised benchmark
│   └── 02_supervised_fusion.ipynb                   # phase 2 — supervised + fusion
├── src/
│   └── run_supervised.py                       # phase-2 pipeline (seed=42, reproducible)
├── data/
│   └── README.md                               # how to fetch the dataset (not versioned)
└── docs/                                        # deliverables & rendered outputs
    ├── notebook_original.ipynb                 # original working notebook
    ├── presentation.pptx                       # defence slides
    ├── *.html                                  # rendered notebooks (read without Jupyter)
    └── figures/                                # charts used in this README
```

### Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# fetch the dataset — see data/README.md
jupyter lab notebooks/01_product_classification_text_image.ipynb
```

> **Note.** Raw images (~1,050 files), the USE model (~960 MB) and the virtual
> environment are **not** versioned — see [`data/README.md`](data/README.md) to
> fetch them. The showcase notebook is written to be *read*; the full original
> run lives in [`docs/`](docs/).

### Limits & next steps

- **Late fusion** — combine per-modality probabilities (averaging / stacking)
  instead of concatenating raw features, which lost accuracy here.
- Fine-tune BERT and a vision backbone end-to-end instead of frozen features.
- Target the hard pairs surfaced by the confusion matrices (*Home Decor* vs
  *Home Furnishing*).
- Scale beyond 1,050 products to confirm the ~0.82 ceiling holds.

---

## 🇫🇷 Version française

### La question

Une marketplace e-commerce repose sur des catégories saisies à la main par les
vendeurs — lent, incohérent, source d'erreurs. **Peut-on attribuer la catégorie
automatiquement** à partir de ce qu'on a déjà : une description texte et une
photo ? Ce projet y répond *avant* de construire un classifieur de production,
en mesurant la quantité de signal « catégorie » que porte chaque source.

### Démarche — un benchmark non supervisé, honnête

Plutôt que de foncer sur un modèle supervisé, j'ai traité le sujet comme une
**étude de faisabilité**. Le même protocole est appliqué à chaque jeu de
features, pour que les chiffres soient directement comparables :

```
features → PCA (débruitage) → t-SNE (2D) → K-Means (k=7) → Adjusted Rand Index
```

L'**Adjusted Rand Index (ARI)** mesure l'accord entre les 7 clusters non
supervisés et les 7 vraies catégories (0 = aléatoire, 1 = parfait). Aucun label
ne sert à construire les features — seulement à les évaluer.

### Résultats

| Modalité | Méthode | ARI |
|---|---|:--:|
| Texte | CountVectorizer (sac de mots) | 0,333 |
| Texte | TF-IDF | 0,240 |
| **Texte** | **Word2Vec** | **0,475** 🏆 |
| Texte | BERT (`bert-base-uncased`, [CLS]) | 0,182 |
| Texte | BERT (TF-Hub) | 0,289 |
| Texte | USE (Universal Sentence Encoder) | 0,423 |
| Image | SIFT (sac de mots visuels) | 0,053 |
| **Image** | **CNN — VGG16 transfer learning** | **0,460** 🏆 |

![Benchmark ARI des 8 méthodes](docs/figures/ari_benchmark.png)

Le meilleur modèle image, VGG16, mérite un zoom. Le clustering seul donne
ARI 0,460 ; une fois les clusters alignés de façon optimale sur les vraies
catégories (affectation hongroise), cela correspond à **66% d'accuracy** — sur
7 classes, à partir des images seules, sans supervision. La matrice de confusion
montre que le modèle confond exactement les catégories qu'un humain confondrait :
*Home Furnishing* déborde sur *Baby Care* et *Home Decor* (textiles et déco
visuellement proches), tandis que *Watches* est quasi parfait (138/150).

![Matrice de confusion VGG16](docs/figures/cnn_confusion_matrix.png)

### Lecture des résultats

- **Faisabilité : oui.** Les meilleures méthodes texte (0,475) et image (0,460)
  retrouvent les catégories bien au-dessus du hasard — sans aucun étiquetage
  manuel.
- **Le simple bat le lourd sur ces données.** Word2Vec dépasse BERT (0,18–0,29).
  Les descriptions produits sont des sacs de mots-clés, pas des phrases : la force
  contextuelle de BERT, utilisé gelé, ne sert à rien. *On choisit le modèle selon
  la donnée, pas selon la mode.*
- **Le transfer learning débloque l'image.** VGG16 (0,460) vs SIFT (0,053) — un
  gain d'environ 9×. Les features apprises se transfèrent, pas les features
  artisanales.
- **Les deux modalités sont complémentaires** (texte ≈ image) — fusion testée en
  phase 2 ci-dessous.

### Phase 2 — classification supervisée & fusion

La phase 1 répond à *« le signal existe-t-il ? »*. La phase 2 répond à
*« à quel point sait-on classer ? »* — un classifieur supervisé (régression
logistique), seed figé, split stratifié 80/20, accuracy + macro-F1.

| Features | Accuracy | Macro-F1 |
|---|:--:|:--:|
| **Image — VGG16** | **0,824** | 0,821 |
| **Texte — Word2Vec** | **0,814** | 0,809 |
| Fusion (texte ⊕ image, concat naïve) | 0,781 | 0,776 |

![Accuracy supervisée — modalité seule vs fusion](docs/figures/supervised_accuracy.png)

Deux résultats à énoncer franchement :

- **~0,82 d'accuracy sur 7 classes**, avec une seule modalité — la faisabilité
  devient un chiffre dur, plus une intuition.
- **La fusion naïve fait *moins bien* (0,78).** Concaténer 300 dims texte et
  200 dims image donne au classifieur 500 variables d'échelles et de densités
  inégales ; les dimensions faibles ajoutent du bruit plus vite que du signal.
  *Une fusion est un choix de modélisation, pas un `np.hstack`.* La bonne suite :
  late fusion (combiner les probabilités par modalité) ou un espace joint appris
  — voir [notebook 02](notebooks/02_supervised_fusion.ipynb).

> **Note de reproductibilité.** L'étude originale de 2023 affichait des valeurs
> d'ARI différentes entre les slides et le notebook (ex. USE ≈ 0,49 vs 0,42)
> car les seeds n'étaient pas figés. Tout ici est **ré-exécuté avec `seed=42`
> figé** (Word2Vec, PCA, t-SNE, K-Means, split train/test), donc les chiffres
> sont stables et reproductibles. Le tableau ARI de la phase 1 reflète les
> valeurs du notebook.

### Stack technique

**NLP** — NLTK (tokenisation, stop-words, stemming Porter, lemmatisation
WordNet), scikit-learn (CountVectorizer, TF-IDF), gensim (Word2Vec), Hugging Face
Transformers (BERT), TensorFlow Hub (USE).
**Computer Vision** — OpenCV (SIFT), TensorFlow / Keras (VGG16 transfer learning).
**ML & viz** — scikit-learn (PCA, t-SNE, K-Means, ARI), pandas, matplotlib, seaborn.

### Reproduire

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# récupérer le dataset — voir data/README.md
jupyter lab notebooks/01_product_classification_text_image.ipynb
```

> **Note.** Les images brutes (~1 050 fichiers), le modèle USE (~960 Mo) et
> l'environnement virtuel ne sont **pas** versionnés — voir
> [`data/README.md`](data/README.md). Le notebook vitrine est pensé pour être
> *lu* ; le run original complet est dans [`docs/`](docs/).

### Limites & suites

- **Late fusion** — combiner les probabilités par modalité (moyenne / stacking)
  au lieu de concaténer les features brutes, qui a fait perdre de l'accuracy ici.
- Fine-tuner BERT et un backbone vision de bout en bout plutôt que des features
  gelées.
- Viser les paires difficiles révélées par les matrices de confusion
  (*Home Decor* vs *Home Furnishing*).
- Passer à l'échelle au-delà de 1 050 produits pour confirmer le plafond ~0,82.

---

*Dataset: [Flipkart products](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products) (public, CC0). Project built for the OpenClassrooms Data Scientist path.*
