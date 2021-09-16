# coding: utf-8
# Loading pickle files of nlvr2 for datasets.
# This module consists of two objects, load_dataset and ImageFeatureFormatter.
# An instance of ImageFeatureFormatter is registered with image_feature key 
# to datasets when importing this module.
# Image keys in examples are converted to image features by this formatter each time.
# You can load dataset dict using load_dataset.


from collections import OrderedDict
import pickle
import json
import os
#import numpy as np
import torch
import datasets

from eval_vl_glue import VoltaImageFeature


class ImageFeatureFormatter(datasets.formatting.CustomFormatter):
    
    def __init__(self, dataset_dir=None, model_config=None, transform=None):
        
        assert model_config is not None, 'Specify model_config when you use set_format.'
        
        self.add_global_image_feature = model_config.add_global_imgfeat
        
        assert model_config.num_locs in (4, 5), f'not supported num_loc: {model_config.num_locs}'
        self.num_locs = model_config.num_locs
        self.add_area = (model_config.num_locs == 5)
        
        # We consider model_config.default_num_boxes is the maximum sequence size
        self.num_boxes_per_image = model_config.default_num_boxes
        if self.add_global_image_feature:
            self.num_boxes_per_image += 1
        self.total_seq_len = 2 * self.num_boxes_per_image
        
        self.v_feature_size = model_config.v_feature_size
        
        self.pickled_file_path = os.path.join(dataset_dir, 'pickled')
        self.transform = transform
    
    def get_feature(self, image_id):
        """Load and return image features related to image_id
        pickle file is defined in the convert_lmdb.py
        ```
        outputs = (
            feature.add_global_image_feature,
            feature.add_area,
            feature.num_boxes,
            feature.image_location.numpy(),
            feature.features.numpy(),
        )
        ```
        """
        split = image_id.split('-')[0]
        file_path = os.path.join(self.pickled_file_path, split, image_id)
        with open(file_path, 'rb') as f:
            return VoltaImageFeature.from_dict(
                pickle.Unpickler(f).load(),
                add_area=self.add_area, 
                add_global_image_feature=self.add_global_image_feature,
            )
    
    def _fit_length(self, feature):
        """
        Returns:
            fitted length, fitted boxes, fitted features
        """
        if feature.num_boxes <= self.num_boxes_per_image:
            return feature.num_boxes, feature.image_location, feature.features
        l = self.num_boxes_per_image
        if self.add_global_image_feature != 'last':
            return l, feature.image_location[:l], feature.features[:l]
        b = torch.cat([feature.image_location[:l-1], feature.image_location[-1:]], axis=0)
        f = torch.cat([feature.features[:l-1], feature.features[-1:]], axis=0)
        return l, b, f
    
    def get_cat_image_feature(self, image_id_0, image_id_1):
        """Concatenate two image feature sets to make an example.
        """
        
        boxes = torch.zeros((self.total_seq_len, self.num_locs), dtype=torch.float32)
        features = torch.zeros((self.total_seq_len, self.v_feature_size), dtype=torch.float32)
        image_mask = torch.zeros((self.total_seq_len,), dtype=torch.int64)
        
        offset = 0
        for image_id in (image_id_0, image_id_1):
            l, b, f = self._fit_length(self.get_feature(image_id))
            boxes[offset:offset+l] = b
            features[offset:offset+l] = f
            image_mask[offset:offset+l] = 1
            offset += self.num_boxes_per_image
        
        return {
            'input_images': features,
            'image_loc': boxes,
            'image_attention_mask': image_mask,
         }
    
    def _preprocess(self, examples):
        """Add image features to examples
        """
        updates = [self.get_cat_image_feature(i, j) for i, j in zip(examples['image_id_0'], examples['image_id_1'])]
        for key in ('input_images', 'image_attention_mask', 'image_loc'):
            examples[key] = [_[key] for _ in updates]
        del examples['image_id_0'], examples['image_id_1'], 
        return examples
        
    def format_batch(self, pa_table):
        batch = self.python_arrow_extractor().extract_batch(pa_table)
        batch = self._preprocess(batch)
        if self.transform is None:
            return batch
        return self.transform(batch)

# Add this formatter to datasets
datasets.formatting._register_formatter(ImageFeatureFormatter, 'image_feature')


# Make examples form annotation
# This key conversion is required to match Datasets' notation.
def example_mapping(src_example):
    
    prefix = '-'.join(src_example['identifier'].split('-')[:-1])
    label = -1 if src_example['label'] is None else (1 if src_example['label'] != 'True' else 0)
    
    dest = OrderedDict()
    dest['id'] = src_example['identifier']
    dest['image_id_0'] = prefix+'-img0'
    dest['image_id_1'] = prefix+'-img1'
    dest['sentence'] = src_example['sentence']
    dest['label'] = label
    return dest


def does_features_exist(pickled_file_dir, example):
    
    for image_id in (example['image_id_0'], example['image_id_1']):
        split = image_id.split('-')[0]
        file_path = os.path.join(pickled_file_dir, split, image_id)
        if not os.path.exists(file_path):
            return False
    return True
                    

def load_dataset_vl(dataset_dir=None, filter_by_features=True):
    """
    Load a DatasetDict object for this dataset.
    The object will have three keys; train, validation and test.
    This dataset uses an image feature transform named image_feature.
    Please apply that format after all preprocessing for the dataset end.
    ```
    ...
    dataset = load_dataset()
    ...
    (preprocessing such as dataset.map)
    ...
    datasets.set_format(type='image_feature', model_config=config)
    ...
    (run training)
    ```
    If filter_by_features is True, we remove annotations that do not have the features.
    """
    
    split_mapping = {
        'dev': 'validation',
        'test0': 'test'
    }
    
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    pickled_file_dir = os.path.join(dataset_dir, 'pickled')
    
    # Load a json file
    def _load_from_json(file_path):
        
        examples = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                example = example_mapping(json.loads(line))
                if filter_by_features and not does_features_exist(pickled_file_dir, example):
                    continue
                examples.append(example)
        if examples:
            keys = list(examples[0].keys())
            return datasets.Dataset.from_dict({k:list(_[k] for _ in examples) for k in keys})
        return datasets.Dataset.from_dict({})
    
    # Make each split and uniy them into a DatasetDict
    dataset_dict = datasets.DatasetDict()
    for name in os.listdir(annotations_dir):
        if name.endswith('.json'):
            path = os.path.join(annotations_dir, name)
            raw_label = os.path.splitext(name)[0]
            label = split_mapping.get(raw_label, raw_label)
            dataset_dict[label] = _load_from_json(path)
    
    return dataset_dict
