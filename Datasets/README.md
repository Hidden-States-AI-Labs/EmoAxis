# Datasets

This directory contains the datasets used in this project for **multi-label emotion classification**. All datasets have been **preprocessed and standardized** to ensure consistency, reproducibility, and seamless integration with transformer-based deep learning models.

---

## 📂 Dataset Overview

The following benchmark emotion datasets are included:

* **GoEmotions**
* **EmoPillars (Context-less, Full Version)**

Each dataset is organized into the following splits:

* **Train**
* **Validation**
* **Test**

These splits are maintained separately for each dataset to enable fair evaluation and reproducible experimentation.

---

## 🔧 Preprocessing Details

All datasets in this directory have been **preprocessed by us** prior to model training and inference. The preprocessing pipeline was designed to unify label representations and support multi-label learning across datasets.

### Key preprocessing steps:

* Conversion of original emotion annotations into **multi-one-hot encoded labels**
* Standardization of label dimensions across datasets
* Removal of extraneous metadata and unused columns
* Preservation of semantic and emotional content in the text

The preprocessing is applied **prior to tokenization** and consistently across all datasets unless domain-specific adaptations are required.

---

## 📊 Data Format

Each dataset split (`train`, `validation`, `test`) follows a **uniform tabular structure** with the following columns:

* **`text`**
  The preprocessed textual input (one sample per row).

* **Emotion label columns (`0` to `27`)**
  A total of **28 binary columns**, where each column corresponds to a specific emotion class.

```
text | 0 | 1 | 2 | ... | 27
```

* `1` indicates the presence of the corresponding emotion
* `0` indicates its absence

This structure supports **multi-label classification**, allowing a single text instance to express multiple emotions simultaneously.

---

## 🏷️ Emotion Label Mapping

The column indices (`0–27`) correspond to the following emotion categories.
This mapping is **consistent across GoEmotions and EmoPillars** and applies to all dataset splits.

| Index | Emotion        |
| ----: | -------------- |
|     0 | Admiration     |
|     1 | Amusement      |
|     2 | Anger          |
|     3 | Annoyance      |
|     4 | Approval       |
|     5 | Caring         |
|     6 | Confusion      |
|     7 | Curiosity      |
|     8 | Desire         |
|     9 | Disappointment |
|    10 | Disapproval    |
|    11 | Disgust        |
|    12 | Embarrassment  |
|    13 | Excitement     |
|    14 | Fear           |
|    15 | Gratitude      |
|    16 | Grief          |
|    17 | Joy            |
|    18 | Love           |
|    19 | Nervousness    |
|    20 | Optimism       |
|    21 | Pride          |
|    22 | Realization    |
|    23 | Relief         |
|    24 | Remorse        |
|    25 | Sadness        |
|    26 | Surprise       |
|    27 | Neutral        |

---

## 🧠 Label Encoding Scheme

* Labels are represented using **multi-one-hot vectors**
* Each row may have **multiple active emotion labels**
* The label space dimensionality is fixed at **28 emotions**
* The encoding is compatible with common multi-label loss functions (e.g., Binary Cross-Entropy)

---

## ✅ Intended Use

These datasets are intended for:

* Multi-label emotion classification tasks
* Training and evaluation of transformer-based language models
* Cross-dataset generalization and benchmarking
* Emotion representation learning and analysis

---

## 📚 Citations

If you use these datasets in your research, please cite the original sources:

### GoEmotions

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020).
**GoEmotions: A Dataset of Fine-Grained Emotions.**
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020).*

```bibtex
@inproceedings{demszky2020goemotions,
  title     = {GoEmotions: A Dataset of Fine-Grained Emotions},
  author    = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year      = {2020}
}
```

---

### EmoPillars (Context-less, Full Version)

Shvets, A. (2025).
**Emo Pillars: Knowledge Distillation to Support Fine-Grained Context-Aware and Context-Less Emotion Classification.**
*arXiv preprint arXiv:2504.16856.*

```bibtex
@misc{shvets2025emopillarsknowledgedistillation,
  title        = {Emo Pillars: Knowledge Distillation to Support Fine-Grained Context-Aware and Context-Less Emotion Classification},
  author       = {Alexander Shvets},
  year         = {2025},
  eprint       = {2504.16856},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2504.16856}
}
```

---

## ⚠️ Notes

* Only **preprocessed** versions of the datasets are included
* Raw datasets are not provided in this directory
* Tokenization and model-specific input formatting should be applied downstream
* Any dataset-specific constraints or extensions should be documented at the experiment level

---
