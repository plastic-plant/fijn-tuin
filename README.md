**Fijn-tuin 🏡**  _(Dutch for “nice garden”)_ is a fine-tune on [Mistral 7B](https://mistral.ai), answering questions on gardening club regulation. Trained on base model [GEITJEtje 7B](https://github.com/Rijgersberg/GEITje) by [Edwin Rijgersberg](https://goingdutch.ai/nl/), the merged model returns Dutch language with adjusted weights favouring answers likely to come up in a conversation completing Q&A of an Amsterdam based garden association.

```text
  > python 2-infer.py
```
```text
  Example generated response:

  JIJ:  Hoeveel gasflessen zijn maximaal toegestaan op een volkstuinpark met vergunning om te overnachten?
  LLM:  Op een volkstuinpark met vergunning om te overnachten zijn maximaal 2 gasflessen toegestaan.
```

While generally a fine-tune (on a dataset this size) is not for knowledge injection, here's an example that does just that. The result is a compact model that can infer given information without additional prompting, allowing us to keep context small during inference. Base model is a full-parameter finetune of Mistral on Dutch text. On top of that an adapter is created and saved, and a full merged model is exported. We can swap out the adapter and test, revert to a previous version or push the merged model.

The [LoRA adapter](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) is essentially a set of matrices that are trained to modify the behavior of specific layers in the base model. It acts as an extension to the original model for targeted fine-tuning without altering the entire model's parameters. Efficient for a use case where we want to retrain on datasets more often without resorting to [RAG](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/).
```python
  model = PeftModel.from_pretrained(base_model, './model/adapter')
  
  model/adapter
    - adapter_model.safetensors
```
```python
  model = AutoModelForCausalLM.from_pretrained('./model/full')
  
  model/full
    - model.safetensors
```

Training and merging the adapter with `1-train.py` is fast and reasonable in requirements. The full run and merge topped at 30 GB of memory, took about 100 GB of disk space and fitted a GPU with 22 GB VRAM. As a test I pushed for longer training times.

The NVIDIA L4 with additional disk space and memory offered in [Google Collab Pro](https://colab.research.google.com/signup) gets it done in about 40 minutes for 7 epochs, 10 minutes for 4 epochs and 5 minutes for a single pass. _(Plus, an additional 5 minutes overhead for downloading assets and reading tensors from disk.)_ That's the equivalent of less than a euro in computing units.

Larger batch sizes for bigger datasets benefit from a more comfortable A100. That comes with twice the memory, but doubles costs. For desktops, the NVIDIA RTX 6000 Ada (48 GB) and RTX 4090 (24 GB RAM, but more capable) perform similar.
