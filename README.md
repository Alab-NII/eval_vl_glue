# eval_vl_glue

This is the repository for the paper: Effect of Visual Extensions on Natural Language Understanding in Vision-and-Language Models (EMNLP 2021).
This paper evaluates NLU in some V&L models pre-trained in the [VOLTA framework](https://github.com/e-bug/volta) using [the GLUE Benchmark](https://gluebenchmark.com/).  

We publish the source codes and some weights for our models and GLUE evaluation.

In this README, we describe the outline of this repository.

- **eval\_vl\_glue**: directory for the eval_vl_glue python package that cotains extracter and transformers_volta.
- **vl_models**: directory for pre-trained and fine-tuned models.
- **demo**: Notebooks for demonstration.
- **evaluation**: directory for our evalutaion experiments.
- **download**: directory for downloaded files.

## Advance Preparation

We assume that:
- We can use Python3 from the 'python' command.

    We used the venv of python3.

- pip is upgraded:
    
    ```
    pip install -U pip
    ```

- PyTorch and torchvision appropriate for your environment is installed.

    We used the version of 1.9.0 with CUDA 11.1.

- The Notebook packages are installed when you run notebooks in your environment:

    ```
    pip install notebook ipywidgets
    jupyter nbextension enable --py widgetsnbextension
    ```

## How to Reproduce Our Experiments

1. **Clone this repository.**

2. **Install the eval_vl_glue package to use transformers_volta.**

    ```
    pip install -e .
    ```
    
    transformers_volta provides an interface similar to Huggingface's Transformers for V\&L models in the Volta framework.  
    See [the transformers\_volta section in eval\_vl\_glue](/eval_vl_glue#transformers_volta) for more detail.

3. **Prepare pretrained models for transformers_volta.**
    
    We describe the way to obtain those models in [vl_models](/vl_models).
    
4. **Fine-tune the models with evaluation/glue_tasks/run_glue.py .**

    The run_glue.py script is a script to fine-tune a model on the GLUE task.
    We modified [run_glue.py](https://github.com/huggingface/transformers/blob/v4.4.0/examples/text-classification/run_glue.py) in the Huggingface's transformers repository, and the usage is basically the same as the original one.  
    See [the glue_tasks section in evaluation](/evaluation#glue_tasks).

5. **Summarize the results.**

    We used Notebook to summarize the results (get_glue_score.ipynb) .  
    See [the analysis section in the evaluation](/evaluation#analysis).

## Quick Check for Our Implementation and Conversion

We checked our implementation and conversion briefly in the following ways:

1. **Training pre-trained models on the V\&L task to compare to the original Volta.**

    See [the evaluation vl_tasks in the evaluation](/evaluation#vl_tasks) for the results.

2. **Masked token prediction with image context.**

    See [demo/masked_lm_with_vision_demo.ipynb](/demo/masked_lm_with_vision_demo.ipynb) for the results.

## PyTorch Image Extractor

The original Volta framework relies on an image detector pretrained in [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998) ([Github repository](https://github.com/peteanderson80/bottom-up-attention)).
This detector runs with a specific version of the caffe framework (typically configured in a docker environment).

To improve the connectivity to PyTorch models, we converted the part for image extraction of this model, including model definition, weight and detection procedure, into a PyTorch model.
You can access the converted model from extractor of the eval_vl_glue package.  
See [the extractor section in eval_vl_glue](/eval_vl_glue#extractor) for more detail.

![Comparision of detected regions](/download/comparison.jpg)

Notes:
- This extractor work was completed after our paper, so our results in the paper was based on the original detector. 
- The outputs of the converted detector are similar, but not fully identical to those of the original model as you can see in the above figure (see [demo/extractor_demo.ipynb](/demo/extractor_demo.ipynb) and [demo/original_vs_ours.ipynb](/demo/original_vs_ours.ipynb) for more detail).
- We have not conducted quantitative bench marking.

## License

This work is licensed under the Apache License 2.0 license. 
See LICENSE for details. 
Third-party software and data sets are subject to their respective licenses.  
If you find our work useful in your research, please consider citing the paper:

CITATION

## Acknowledgement

We created our code with reference to the following repository:
- https://github.com/peteanderson80/bottom-up-attention
- https://github.com/airsplay/lxmert
- https://github.com/huggingface/transformers
- https://github.com/e-bug/volta

We also use the pre-trained weights available in the following repository:
- https://github.com/peteanderson80/bottom-up-attention
- https://github.com/e-bug/volta

We would like to thank them for making their resources available.
