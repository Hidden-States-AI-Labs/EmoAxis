
## Model Evaluation Options

You can evaluate the model using **either** of the following approaches:

---
### Using `AutoModel` from 🤗 Transformers

You can load the model directly from Hugging Face using `AutoModel` and test it on your own data.

* **Folder:** `Model-Evaluate`

#### Important Notes

* Use `trust_remote_code=True` in `AutoModel.from_pretrained` to support our custom model class **EmoAxis** based on **RoBERTa-architecture**.
* To avoid unnecessary warnings during model download, you can suppress Transformers logs.

---

---




