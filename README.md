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

### What the numbers say

- **Feasibility: yes.** The best text (0.475) and image (0.460) methods both
  recover categories well above chance — with zero manual labelling.
- **Simpler beat heavier on this data.** Word2Vec outscored BERT (0.18–0.29).
  Product descriptions are keyword bags, not sentences, so BERT's contextual
  strength goes unused when frozen. *Match the model to the data, not to the hype.*
- **Transfer learning is the unlock for images.** VGG16 (0.460) vs SIFT (0.053) —
  a ~9× gain. Learned features transfer across domains; hand-crafted ones don't.
- **The two modalities are complementary** (text ≈ image), which is the case for
  fusing them in the next step.

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
├── notebooks/
│   └── 01_product_classification_text_image.ipynb   # clean, readable case study
├── data/
│   └── README.md                               # how to fetch the dataset (not versioned)
└── docs/                                        # OpenClassrooms deliverables
    ├── notebook_original.ipynb                 # original working notebook
    └── presentation.pptx                       # defence slides
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

- Unsupervised feasibility only — the natural follow-up is a **supervised**
  classifier reporting accuracy / F1 per category.
- **Fuse** the best text (Word2Vec / USE) and image (VGG16) features and measure
  the lift over each single modality.
- Fine-tune BERT and a vision backbone end-to-end instead of frozen features.
- Target the hard pairs surfaced by the confusion matrix (*Home Decor* vs
  *Home Furnishing*).

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
- **Les deux modalités sont complémentaires** (texte ≈ image) — ce qui justifie
  leur fusion à l'étape suivante.

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

- Faisabilité non supervisée uniquement — la suite logique est un classifieur
  **supervisé** avec accuracy / F1 par catégorie.
- **Fusionner** les meilleures features texte (Word2Vec / USE) et image (VGG16)
  et mesurer le gain sur chaque modalité seule.
- Fine-tuner BERT et un backbone vision de bout en bout plutôt que des features
  gelées.
- Viser les paires difficiles révélées par la matrice de confusion (*Home Decor*
  vs *Home Furnishing*).

---

*Dataset: [Flipkart products](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products) (public, CC0). Project built for the OpenClassrooms Data Scientist path.*
