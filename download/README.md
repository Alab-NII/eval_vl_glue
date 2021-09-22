# eval\_vl\_glue/download

Directory for downloaded files.

- **volta_config** contains the configuration files for controlled models; distributed [here](https://github.com/e-bug/volta/tree/main/config).

- **volta_weights** is the directory to place the original volta weights for the model conversion.

- **image files** are images used for our demo notebooks.

## Extractor's weight

Our repository assumes that weights for the extractor are placed in this directory.  
See [the extractor section in eval_vl_glue](/eval_vl_glue#extractor) for the detail.

## volta_weights

In this work, we added the Huggingface's transformers' like interface to volta models.  
To convert the original volta weights, first download the original weights (and configs if [/download/volta_config](/download/volta_config) does not contain them) from the [e-bug/volta](https://github.com/e-bug/volta/blob/main/MODELS.md) repository and put them in this directory.
Second run the conversion script:

```
cd <path to the repository's root>
python vl_models/convert.py -c <path to the config file> -w <path to the weight file>
```

This will make a converted model in the vl_models/pretrained directory.

The detail is available at [vl_models](/vl_models#weights-for-the-transformers_volta-models)

## Source of the image files

We cited those images from [the PASCAL Visual Object Classes Challenge 2007](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/) test sets except for the png files.
To see detailed information about each image (such as the ownership), refer the annotated test data available in the web site. 
