# eval_vl_glue/transformers_volta

Transformers used for our experiments.

We made this module based on transformers (https://github.com/huggingface/transformers).

We customized 4.4.0.dev0　to leave only the parts related to our experiment.
- We left only models we used in our experiments and models required by those models: auto, BERT, encoder_decoder
- We added a volta model (in the models/volta directory).
- Some files for the model registration (such as __init__.py) have been changed.

We used codes in https://github.com/e-bug as a reference for the volta model.


## How to use

When you install this repository with the editable mode of pip, you can load models like the original transformers:

```
from eval_vl_glue import transformers_volta

name_or_path_to_a_model = 'conversion/hf_volta_models/ctrl_vilbert_base'
# path may change depending on where you made the models

model = transformers_volta.AutoModel.from_pretrained(name_or_path_to_a_model)
tokenizer = transformers_volta.AutoTokenizer.from_pretrained(name_or_path_to_a_model)
```

Currently, we have prepared classes for AutoModel and AutoModelForSequenceClassification.
