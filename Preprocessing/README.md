````markdown
# Emotion Dataset Preprocessing

This repository contains preprocessing code used to prepare three emotion classification datasets for experimentation and model training:

- **GoEmotions**
- **SemEval 2018 Task 1 (English)**
- **EmoPillars (Contextless-Full)**

The scripts handle dataset loading, cleaning, formatting, and (where applicable) downsampling to ensure consistency across datasets.

---

## Datasets

### 1. GoEmotions

**GoEmotions** is a large-scale dataset of English Reddit comments annotated with fine-grained emotion labels.

- **Source**: Hugging Face `datasets` library  
- **Download method**: Programmatically loaded using `load_dataset`
- **Notes**: No manual download is required

---

### 2. SemEval 2018 Task 1 (English)

**SemEval 2018 Task 1: Affect in Tweets** is a benchmark dataset for emotion and sentiment analysis in tweets.

- **Source**: Official SemEval website
- **Download method**: Manual download
- **URL**:  
  https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip

After downloading, the raw files are processed using the preprocessing scripts provided in this repository.

---

### 3. EmoPillars (Contextless-Full)

**EmoPillars** is a fine-grained emotion dataset supporting both context-aware and context-less emotion classification.

- **Subset used**: `contextless-full`
- **Source**: Hugging Face
- **Download method**: Manual download
- **URL**:  
  https://huggingface.co/datasets/alex-shvets/EmoPillars/tree/main/context-less

The dataset is further **preprocessed and downsampled** using the code in this repository.

---

## Preprocessing

The preprocessing scripts are designed to:

- Normalize and clean text
- Convert labels into a unified format
- Remove incomplete or invalid samples
- Downsample data (EmoPillars)
- Export processed datasets for downstream training and evaluation etc..

Each dataset has its own dedicated preprocessing script.

---
## Citations

If you use any of the original datasets , please cite the following papers:

### SemEval 2018 Task 1

```bibtex
@inproceedings{SemEval2018Task1,
  author    = {Mohammad, Saif M. and Bravo-Marquez, Felipe and Salameh, Mohammad and Kiritchenko, Svetlana},
  title     = {SemEval-2018 {T}ask 1: {A}ffect in Tweets},
  booktitle = {Proceedings of the International Workshop on Semantic Evaluation (SemEval-2018)},
  address   = {New Orleans, LA, USA},
  year      = {2018}
}
```

### GoEmotions

```bibtex
@inproceedings{demszky2020goemotions,
  title     = {GoEmotions: A Dataset of Fine-Grained Emotions},
  author    = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year      = {2020}
}
```

### EmoPillars

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

## License and Dataset Usage

This repository only provides **preprocessing code**.
Please refer to the original dataset licenses and terms of use before redistribution or commercial use.

---

## Acknowledgements

We thank the authors of **GoEmotions**, **SemEval 2018 Task 1**, and **EmoPillars** for making their datasets publicly available.

```

---
