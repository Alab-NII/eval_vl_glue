# coding: utf-8
# Feature holder class and its utilities.


from dataclasses import dataclass

import base64
import torch
import numpy as np


@dataclass
class VoltaImageFeature:
    
    add_global_image_feature: str
    add_area: bool
    num_boxes: int
    image_location_ori: torch.Tensor
    image_location: torch.Tensor
    features: torch.Tensor
    
    @staticmethod
    def from_regions(regions,  add_area=True, add_global_image_feature='first'):
        """
        Converts a detection result () on an image into tensors.
        Arguments:
            regions: a list of DetectedRegions
            add_area: if ture output location contains the area attribute at the last element
            add_global_image_feature: where to attach 
                a global image feature in an image input seuence ('first', 'last', None)
        Returns:
            a VoltaImageFeature
        """
        # we assume that the original image size is the same among all regions
        image_h = int(regions[0].image_height)
        image_w = int(regions[0].image_width)
        boxes = np.asarray([region.box for region in regions])
        features = np.asarray([region.feature for region in regions])
        
        return VoltaImageFeature._convert_to_tensor(
            image_h, image_w, boxes, features, add_area, add_global_image_feature
        )
        
    @staticmethod
    def from_dict(detection, add_area=True, add_global_image_feature='first'):
        """
        Converts a detection result (dict) on an image into tensors.
        Arguments:
            detection: a dict that contains {'img_h', 'img_w', 'boxes', 'features'}
            add_area: if ture output location contains the area attribute at the last element
            add_global_image_feature: where to attach 
                a global image feature in an image input seuence ('first', 'last', None)
        Returns:
            a VoltaImageFeature
        """        
        image_h = int(detection["img_h"])
        image_w = int(detection["img_w"])
        boxes = detection['boxes'].reshape(-1, 4)
        num_boxes = boxes.shape[0]
        features = detection["features"].reshape(num_boxes, -1)
        
        return VoltaImageFeature._convert_to_tensor(
            image_h, image_w, boxes, features, add_area, add_global_image_feature
        )
        
    @staticmethod
    def _convert_to_tensor(image_h, image_w, boxes, features, 
                          add_area=True, add_global_image_feature='first'):
        
        num_boxes = boxes.shape[0]
        
        # image_location_ori
        # Order of location is (x, y, xx, yy, [area])
        # Area is added if add_area is True
        image_location_ori = np.zeros((num_boxes, 4 + add_area), dtype=np.float32)
        image_location_ori[:, :4] = boxes
        if add_area:
            image_location_ori[:, 4] = ((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
        
        # image_location
        # While image_location_ori is not normalized, image_location is normalized
        image_w_f = float(image_w)
        image_h_f = float(image_h)
        image_location = image_location_ori.copy()
        image_location[:, 0] /= image_w_f
        image_location[:, 1] /= image_h_f
        image_location[:, 2] /= image_w_f
        image_location[:, 3] /= image_h_f
        if add_area:
            image_location[:, 4] /= (image_w_f * image_h_f)
        
        # global image feature
        if add_global_image_feature in ('first', 'last'):
            # We consider the averaged feature on all boxes as global image feature 
            num_boxes = num_boxes + 1
            g_feat = np.mean(features, axis=0)[None]
            g_location_ori = np.asarray(
                [[0, 0, image_w, image_h] + [image_w * image_h] * int(add_area)], dtype=np.float32)
            g_location = np.asarray([[0, 0, 1, 1] + [1] * int(add_area)], dtype=np.float32)
       
            if add_global_image_feature == "first":
                features = np.concatenate([g_feat, features], axis=0)
                image_location_ori = np.concatenate([g_location_ori, image_location_ori], axis=0)
                image_location = np.concatenate([g_location, image_location], axis=0)
            
            elif add_global_image_feature == "last":
                features = np.concatenate([features, g_feat], axis=0)
                image_location_ori = np.concatenate([image_location_ori, g_location_ori], axis=0)
                image_location = np.concatenate([image_location, g_location], axis=0)
        
        return VoltaImageFeature(
            add_global_image_feature = add_global_image_feature,
            add_area = add_area,
            num_boxes = num_boxes,
            image_location_ori = torch.Tensor(image_location_ori),
            image_location = torch.Tensor(image_location),
            features = torch.Tensor(features),
        )
    

# Utilities for detection tsv files
TSV_FIELD_DEF = [
    ("img_id", 'str'), 
    ("img_h", 'int'),
    ("img_w", 'int'),
    ("objects_id", 'np.int64'),
    ("objects_conf", 'np.float32'),
    ("attrs_id", 'np.int64'),
    ("attrs_conf", 'np.float32'),
    ("num_boxes", 'int'),
    ("boxes", 'np.float32'),
    ("features", 'np.float32'),
]


def load_tsv(tsv_path):
    """
    Load a TSV file made by the extract_coco.py script.
    Returns a dict whose keys are img_id and 
        values are detection results (dict)
    """
    def _decode(field_type, val):
        if field_type == 'int':
            return int(val)
        if field_type.startswith('np.'):
            val = np.frombuffer(
                base64.b64decode(val.encode('ascii')), 
                dtype=np.dtype(field_type[3:])
            )
            if field_type == 'np.int64':
                val = val + 1
                # +1 accounts for the __background__ class id
            return val
        return val

    feature_dict = {}
    with open(tsv_path, 'r') as f:
        for line in f.readlines():
            row = {d[0]: _decode(d[1], v) for d, v in zip(TSV_FIELD_DEF, line.split('\t'))}
            feature_dict[row['img_id']] = row
    return feature_dict
