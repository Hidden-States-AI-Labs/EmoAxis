## Model Evaluation Options

You can evaluate the model using **either** of the following approaches:

---
### 1. Using `AutoModel` from 🤗 Transformers

You can load the model directly from Hugging Face using `AutoModel` and test it on your own data.

* **Folder:** `Model-Evaluate`

#### Important Notes

* Use `trust_remote_code=True` in `AutoModel.from_pretrained` to support the custom **RoBERTa-based architecture**.
* To avoid unnecessary warnings during model download, you can suppress Transformers logs.

---

### 2. Using the `.pt` Model File

Alternatively, follow the provided code for accessing and evaluating the `.pt` model.

* **Folder:** `PT-Model-Evaluate`

This option is recommended if you already have the exported PyTorch model file and want direct control over loading and evaluation.

---




