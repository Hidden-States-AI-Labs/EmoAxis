# Model Evaluation Options

You can evaluate the model using either of the following approaches.

## Using `AutoModel` from 🤗 Transformers

The model can be loaded directly from Hugging Face using `AutoModel` and evaluated on your own data.

**Folder:** `Model-Evaluate`

### Important Notes

* Use `trust_remote_code=True` in `AutoModel.from_pretrained` to enable support for our custom model class **EmoAxis**, which is based on a RoBERTa architecture.
* To avoid unnecessary warnings during model download, you may suppress Transformers logging output.

---

## Citation

If you use our model or dataset in your research, please cite the following paper:

```bibtex
@article{DualObjectivesEmotion2026,
  author  = {Karmakar, Arnab and Bera, Subinoy},
  title   = {Do We Need a Classifier? Dual Objectives Go Beyond Baselines in Fine-Grained Emotion Classification},
  journal = {ResearchGate},
  year    = {2026},
  doi     = {10.13140/RG.2.2.16084.46728},
  url     = {https://www.researchgate.net/publication/399329430_Do_We_Need_a_Classifier_Dual_Objectives_Go_Beyond_Baselines_in_Fine-Grained_Emotion_Classification}
}
```

---

