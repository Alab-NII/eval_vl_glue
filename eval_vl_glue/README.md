# eval\_vl\_glue/eval\_vl\_glue

Directory for the eval_vl_glue python package.

The eval_vl_glue package exposes extractor and transformers_volta as the sub-packages.  
To install eval_vl_glue, run pip install *from the repository root directory*.

```
pip install -e .
```

- **extractor** contains the PyTorch definition of the image detector.
- **transformers_volta** provides volta models with Huggingface's transformers' like interface.
- **volta_image_feature.py** defines how to change extracted image features for the volta models.

## extractor

An image feature extractor.

- **rcnn.py**. We combined [the neural network definition](https://github.com/peteanderson80/bottom-up-attention/blob/master/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt) and [the detection procedure](https://github.com/peteanderson80/bottom-up-attention/tree/master/lib) for [peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)'s caffe model to make our PyTorch version.
We also referred [airsplay/lxmert](https://github.com/airsplay/lxmert) for the default values of the detection procedure.  
- **attributes_vocab.txt and objects_vocab.txt**. The class labels defined [here](https://github.com/peteanderson80/bottom-up-attention/tree/master/data/genome/1600-400-20).

### Usage

```
# Load the detector

model_path = '../download/resnet101_faster_rcnn_final_iter_320000.pt' # depends on the download condition.
from eval_vl_glue.extractor import BUTDDetector
extractor = BUTDDetector()
extractor.load_state_dict(torch.load(model_path))
extractor =  extractor.eval()

# Detect from a PIL Image

image_path = '../download/000542.jpg'
import PIL.Image
extractor.detect(PIL.Image.open(image_path))
# returns a list of DetectedRegion.
```

For the detail of the weight, see README.md in the download directory.  
Some demo notebooks are available in the demo directory.

## transformers_volta

Transformers customized for our experiments.

- Base repository is [transformers 4.4.0.dev0](https://github.com/huggingface/transformers/tree/v4.4.0).
- **models/volta**. We added a volta model by using the codes in [e-bug/volta](https://github.com/e-bug/volta) as a reference.
- We keep only models we used in our experiments and models required by those models: auto, BERT, encoder_decoder.
  Some files for the model registration, such as init.py, have been changed.  

### Usage

```
# Load the transformer model (we try the masked token prediction)

from eval_vl_glue import transformers_volta
model_path = '../vl_models/pretrained/ctrl_vilbert'
tokenizer = transformers_volta.AutoTokenizer.from_pretrained(model_path)
model = transformers_volta.AutoModelForMaskedLM.from_pretrained(model_path)
model = model.eval()

# Load the detector model

from eval_vl_glue.extractor import BUTDDetector
model_path = '../download/resnet101_faster_rcnn_final_iter_320000.pt'
extractor = BUTDDetector()
extractor.load_state_dict(torch.load(model_path))
extractor =  extractor.eval()

# Detect and make the image inputs

from eval_vl_glue import VoltaImageFeature 
import PIL.Image
image_path = '../download/000542.jpg'
image_features = extractor.detect(PIL.Image.open(image_path))
x_image = VoltaImageFeature.from_regions(image_features)

# Make the text inputs

text = 'I can see a [MASK].'
x_text = tokenizer(text)

# Predict

import torch
with torch.no_grad():
    outputs = model(
        input_ids=torch.tensor(x_text['input_ids'])[None], 
        input_images=x_image.features[None], 
        image_attention_mask=x_image.image_location[None]
    )
    # None accounts for batch axis
mask_pos = x_text['input_ids'].index(tokenizer.mask_token_id)
tokenizer.convert_ids_to_tokens(outputs.logits[0, mask_pos].argsort(descending=True)[:10].numpy())
```

**Models' forward functions**. 

- We added input_images and image_attention_mask to the forward functions to input image features.
- We have currently omitted position_ids, inputs_embeds and head_mask from the original keyword arguments.

**Default image features**.

Models use their default image features if image features are not provided.  
The default of the default image features are set the features from a black monochrome image.
You can change them by set_default_image_feature of the VoltaModel.

```
# We assume that detector and transformer are the same as those in the above usage.
...
model = transformers_volta.AutoModelForMaskedLM.from_pretrained(model_path)
...

image_features = extractor.detect(PIL.Image.open(image_path))
x_image = VoltaImageFeature.from_regions(image_features)
model.volta.set_default_image_feature(x_image)
```

Note that the number of boxes (default_num_boxes), the number of location format (num_locs) and feature dimension (v_feature_size) of the given image features are required to match with the values in the model configuration.
The default values are 36 + 1 (global features), 5 and 2048 respectively.

