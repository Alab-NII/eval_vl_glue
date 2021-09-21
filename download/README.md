# eval\_vl\_glue/download

Directory for downloaded files.

- **volta_config** contains the configuration files for controlled models; distributed [here](https://github.com/e-bug/volta/tree/main/config).

- **volta_weights** is the directory to place the original volta weights for the model conversion. See the sub-section below.

- **image files** are images used for our demo notebooks.

## Detector Weights

We converted the caffe models described in [peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention#demo) to the PyTorch models to improve portability.  
We share those converted weights in the following links.
Those weights are expected to exist in this download directory.

| file | url |
| ---- | --- |
| resnet101_faster_rcnn_final.pt | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=ESOlEhST7TZBtsOAl_RlUFEBy9Dd81uzlTmne8qGTmvY4w |
| resnet101_faster_rcnn_final_iter_320000.pt | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=ESlZcYHqMGJMu7XAj0vakhkBPPT7cUcVaclATGoH77wDog |

See notebooks in the demo directory for the usage of the detector.

## volta_weights

In this work, we added the Huggingface's transformers' like interface to volta models.
When converting the original volta weights to the models with the interface, download the original weights (and configs if volta_config does not contain them) from the [e-bug/volta repository](https://github.com/e-bug/volta/blob/main/MODELS.md) to this directory and run the script for conversion.

```
cd <path to the repository's root>
python vl_models/convert.py -c <path to the config file> -w <path to the weight file>
```

This will make a converted model in the vl_models/pretrained directory.


## Source of the image files

We cited those images from [the PASCAL Visual Object Classes Challenge 2007](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/) test sets except for the png files.
To see detailed information about each image (such as the ownership), refer the annotated test data available in the web site. 

