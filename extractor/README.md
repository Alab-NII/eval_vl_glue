# eval_vl_glue/extractor

This directory contains an image extractor for our experiment.

We used an image detector pretrained by Peter Anderson et al, (https://github.com/peteanderson80/bottom-up-attention).  
We converted their caffe model into a torch module.

## Items

### detect_demo.ipynb

See how to use our module in the detect_demo.ipynb.

### rcnn.py

rcnn.py is model definition created in a semi-automatic way from the original caffe prototxt.

### data

Files required to use the image extractor is in the data directory except for the model weights.

For the model weights, please download from here:
- resnet101_faster_rcnn_final.pt : https://drive.google.com/file/d/1frQmoq8MkSaed1RIQepb4qpOYJ2_Dr2N/view?usp=sharing
- resnet101_faster_rcnn_final_iter_320000.pt : https://drive.google.com/file/d/15sQinKbn-N-nM7-9Y-3Rzj3N007ymK9y/view?usp=sharing
