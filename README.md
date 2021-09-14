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

And please install PyTorch and torchvision with an appropriate version for your environment.


## Usage

### Install

#### 1. Clone this repository.

#### 2. Install the eval_vl_glue package codes.

We combined extractor and transformers_volta into a package eval_vl_glue.
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

#### 3. Download weights for the extractor.

Download the weight files from links below manually:
- resnet101_faster_rcnn_final_iter_320000.pt : https://drive.google.com/file/d/15sQinKbn-N-nM7-9Y-3Rzj3N007ymK9y/view?usp=sharing

You can place those files anywhere you want.
We assume that they are in the download directory in the following description.


### Model Conversion

TODO

You can see some details for extractor and transformers_volta in the README.md in each directory.

### Evaluation 

TODO

## License

This work is licensed under the Apache License 2.0 license. 
See LICENSE for details. 
Third-party software and data sets are subject to their respective licenses.  
If you find our work useful in your research, please consider citing the paper:

CITATION

## Acknowledgement

We created our code with reference to the following repository:
- https://github.com/peteanderson80/bottom-up-attention
- https://github.com/huggingface/transformers
- https://github.com/e-bug/volta

We also use the pre-trained weights available in the following repository:
- https://github.com/peteanderson80/bottom-up-attention
- https://github.com/e-bug/volta

We would like to thank them for making their resources available.
