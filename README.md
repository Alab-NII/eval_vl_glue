# eval_vl_glue

This is the repository for the paper: Effect of Visual Extensions on Natural Language Understanding in Vision-and-Language Models (EMNLP 2021).

This paper evaluates NLU in some V&L models pre-trained in the [VOLTA framework](https://github.com/e-bug/volta) using the GLUE Benchmark.

In this repository, we publish source codes for our V&L models, including an image extractor and transformer-based models, and the GLUE evaluation.

## Items

This repository consists of four parts:
- ./eval_vl_glue/extractor : an image extractor for our models
- ./eval_vl_glue/transformers_volta : a volta framework in a customized transformers
- ./conversion : scripts to make models for transformers_volta from pretrained weights
- ./evaluation : scripts to evaluate those models on the GLUE tasks


## Preliminary

We assume that we can use Python3 from the 'python' command.

And please install PyTorch with an appropriate version for your environment.


## Usage

### Install

We combined extractor and transformers_volta into a package eval_vl_glue.

First install the eval_vl_glue package.
You can use pip command:   
(This will install required packages and register eval_vl_glue in your python environment)
```
cd <repository root>
pip install -e .
```

Then, extractor and transformers_volta will be available in python.
```
from eval_vl_glue import extractor, transformers_volta
```

You can see some details for extractor and transformers_volta in the README.md in each directory.

### Model Conversion

TODO

### Evaluation 

TODO

## License

TODO

## Acknowledgement

We created our code with reference to the following repository:
- https://github.com/peteanderson80/bottom-up-attention
- https://github.com/huggingface/transformers
- https://github.com/e-bug/volta

We also use the pre-trained weights available in the following repository:
- https://github.com/peteanderson80/bottom-up-attention
- https://github.com/e-bug/volta

We would like to thank them for making their resources available.
