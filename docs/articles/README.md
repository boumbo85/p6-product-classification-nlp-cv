# References

Papers that back the design choices in this project. PDFs are **not** hosted here
(arXiv's default licence doesn't grant redistribution); follow the links to read
them on arXiv.

## Multimodal e-commerce product categorization (2025)

**Cross-Platform E-Commerce Product Categorization and Recategorization: A
Multimodal Hierarchical Classification Approach**
Q. Han et al., 2025.

- arXiv: https://arxiv.org/abs/2508.20013
- DOI: https://doi.org/10.48550/arXiv.2508.20013
- Subjects: cs.LG, cs.AI, cs.IR

**Why it's cited here.** On the same task (271,700 products, 40 platforms), the
authors compare early / late / attention fusion over RoBERTa, ViT and CLIP. Their
best setup is **CLIP embeddings with an MLP late-fusion head (hierarchical F1
98.59%)**, beating single-modality baselines — the same direction this project
takes at small scale (CLIP shared space, late fusion over early). They also
report a deployment trade-off: late fusion maximizes accuracy with diverse
training data, while early fusion generalizes better to unseen platforms.

*Summary rephrased for licensing compliance — see the paper for the authoritative
text and numbers.*

### BibTeX

```bibtex
@misc{han2025crossplatform,
  title  = {Cross-Platform E-Commerce Product Categorization and Recategorization:
            A Multimodal Hierarchical Classification Approach},
  author = {Han, Qiwei and others},
  year   = {2025},
  eprint = {2508.20013},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi    = {10.48550/arXiv.2508.20013},
  url    = {https://arxiv.org/abs/2508.20013}
}
```
