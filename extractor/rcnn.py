# coding: utf-8
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Taichi Iki


from dataclasses import dataclass

import numpy as np
import torch
import torchvision


@dataclass
class ModelConfig:
    
    objects_vocab_path: str = 'data/objects_vocab.txt'
    attributes_vocab_path: str = 'data/attributes_vocab.txt'
    anchor_feat_stride: int =16
    anchor_scales:tuple =(4, 8, 16, 32)
    
    IMAGE_SCALES: tuple = (600,)
    MAX_SIZE: int = 1000
    NMS: float = 0.3
    SOFT_NMS: float = 0
    SVM: bool = False
    BBOX_REG: bool = True
    HAS_RPN: bool = False
    PROPOSAL_METHOD: str = 'selective_search'
    RPN_NMS_THRESH: float = 0.7
    RPN_PRE_NMS_TOP_N: int = 6000
    RPN_POST_NMS_TOP_N: int = 300
    RPN_MIN_SIZE: int = 16
    AGNOSTIC: bool = False
    HAS_ATTRIBUTES: bool = False
    HAS_RELATIONS: bool = False
    DEDUP_BOXES: float = 1./16.
    PIXEL_MEANS: tuple = (102.9801, 115.9465, 122.7717)
    EPS: float= 1e-14


@dataclass
class DetectedRegion(object):
    
    box: object
    object_label_id: object
    attribute_label_id: object
    
    object_label_conf: object
    attribute_label_conf: object
    
    object_label: object
    attribute_label: object
        

class ProposalModule(object):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    Not support backprop.
    inputs = (rpn_cls_prob_reshape, rpn_bbox_pred, im_info)
    """
    
    def nms(self, proposals, scores, nms_thresh):
        
        # we add 1 to width and height to match py_cpu_nms
        offset = torch.Tensor([[0, 0, 1, 1]], device=proposals.device).to(proposals.dtype)
        return torchvision.ops.nms(proposals + offset, scores, nms_thresh)
    
    def clip_boxes_to_image(self, proposals, image_shape):
        
        # we subtract 1 from the shape to match clip_boxes 
        h, w = image_shape
        return torchvision.ops.clip_boxes_to_image(proposals, (h - 1, w - 1))
        
    def remove_small_boxes(self, proposals, min_size):
        
        # we subtract 1 from the min_size to match _filter_boxes
        return torchvision.ops.remove_small_boxes(proposals, min_size - 1)
    
    @staticmethod
    def bbox_transform_inv(boxes, deltas):
        
        if boxes.shape[0] == 0:
            return torch.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
        
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros(deltas.shape, dtype=deltas.dtype)
        # x1, y1, x2, y2
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes
     
    @staticmethod
    def generate_anchors(
            base_size=16, ratios=(0.5, 1, 2), scales=(8, 16, 32)
        ):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales wrt a reference (0, 0, base_size -1 , base_size - 1) window.
        """
        
        def _break_anchor(anchor):
            # Return width, height, x center, and y center for an anchor (window).
            w = anchor[2] - anchor[0] + 1
            h = anchor[3] - anchor[1] + 1
            cx = anchor[0] + 0.5 * (w - 1)
            cy = anchor[1] + 0.5 * (h - 1)
            return w, h, cx, cy
        
        def _make_anchors(ws, hs, cx, cy):
            # Given a vector of widths (ws) and heights (hs) around a center
            # (x_ctr, y_ctr), output a set of anchors (windows).
            hw = 0.5*(ws[:, None] - 1)
            hh = 0.5*(hs[:, None] - 1)
            return torch.hstack((cx - hw, cy - hh, cx + hw, cy + hh))
        
        ratios = torch.tensor(ratios)
        scales = torch.tensor(scales)
        
        base_anchor = torch.tensor([1, 1, base_size, base_size]) - 1
        
        # Enumerate a set of anchors for each aspect ratio wrt an anchor.
        w, h, cx, cy = _break_anchor(base_anchor)
        size_ratios = (w * h) / ratios
        ws = torch.round(size_ratios**0.5)
        hs = torch.round(ws * ratios)
        ratio_anchors = _make_anchors(ws, hs, cx, cy)
        
        #  Enumerate a set of anchors for each scale wrt an anchor.
        anchors = []
        for ra in ratio_anchors:
            w, h, cx, cy = _break_anchor(ra)
            anchors.append(_make_anchors(w * scales, h * scales, cx, cy))
        
        return torch.vstack(anchors)
    
    def __init__(self, model_config):
        
        self.model_config = model_config
        self.feat_stride = model_config.anchor_feat_stride
        self.anchor_scales = model_config.anchor_scales
        self.anchors_base = self.generate_anchors(scales=self.anchor_scales)
        self.num_anchors = self.anchors_base.shape[0]
        self.train(False)
    
    def train(self, mode):
        
        if mode == True:
            raise NotImplementedError()
        self.training = mode
    
    def __call__(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info):
        
        return self._propose_torch(rpn_cls_prob_reshape, rpn_bbox_pred, im_info)
    
    def _propose_torch(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info):
        """
        rpn_cls_prob_reshape: Tensor(batch_size, 2*num_anchors, height, width)
        rpn_bbox_pred: Tensor(batch_size, 4*num_anchors, height, width)
        im_info:  Tensor(batch_size, input_width, input_height, scale)
        """
        # Currently batch_size is required to be 1
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        
        pre_nms_topN = self.model_config.RPN_PRE_NMS_TOP_N
        post_nms_topN = self.model_config.RPN_POST_NMS_TOP_N
        nms_thresh = self.model_config.RPN_NMS_THRESH
        min_size = self.model_config.RPN_MIN_SIZE
        
        dtype = rpn_cls_prob_reshape.dtype
        height, width = rpn_cls_prob_reshape.shape[-2:]
        
        # 1. Generate proposals from bbox deltas and shifted anchors
        
        # Enumerate all shifts
        # Note: default indexing is different from numpy
        # so, we input y, x order instead of x, y order
        shift_y, shift_x  = torch.meshgrid(
            torch.arange(0, height) * self.feat_stride,
            torch.arange(0, width) * self.feat_stride,
        )
        shift_x = shift_x.ravel()
        shift_y = shift_y.ravel()
        shifts = torch.vstack((shift_x, shift_y, shift_x, shift_y)).permute(1, 0)
        
        anchors = (self.anchors_base[None] + shifts[:, None]).reshape(-1, 4)
        
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = rpn_bbox_pred
        bbox_deltas = bbox_deltas.permute((0, 2, 3, 1)).reshape(-1, 4)
        if self.training and self.model_config.RPN_NORMALIZE_TARGETS:
            bbox_deltas *= self.model_config.RPN_NORMALIZE_STDS
            bbox_deltas += self.model_config.RPN_NORMALIZE_MEANS

        # Same story for the scores:
        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = rpn_cls_prob_reshape[:, self.num_anchors:]
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A,) where rows are ordered by (h, w, a)
        scores = scores.permute((0, 2, 3, 1)).ravel()
        
        # Convert anchors into proposals via bbox transformations
        proposals = self.bbox_transform_inv(anchors, bbox_deltas)
        
        # 2. clip predicted boxes to image
        proposals = self.clip_boxes_to_image(proposals, im_info[0, :2])
        
        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        # Filter works slightly differently from original code for boxes exist near threshold
        # We assume that this has little impact because we don't use those small regions.
        keep = self.remove_small_boxes(proposals, min_size * im_info[0, 2])
        proposals = proposals[keep]
        scores = scores[keep]
        
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        
        order = torch.Tensor(np.argsort(scores.cpu().numpy())[::-1].copy()).to(torch.long)
        #order = scores.argsort(descending=True)
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order]
        scores = scores[order]
        
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = self.nms(proposals, scores, nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep]
        scores = scores[keep]
        
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = torch.zeros((proposals.shape[0], 1), dtype=dtype)
        rois = torch.hstack((batch_inds, proposals.to(dtype)))
        
        return rois, scores


class BUTDDetector(torch.nn.Module):
    
    @staticmethod
    def convert_image_to_tensor(image, output_size_min_side, max_size=None, pixel_means=None):
        """
        arguments:
            image: PIL.Image
            output_size_min_side: int
            max_size: int
            pixel_means: tuple or tensor; BGR order
        returns:
            a reshaped BGR tensor with the shape of (1, 3, height, width)
            a image info (a resized height, a resized width, the scaled used for resizing)
        """
        
        if not isinstance(image, np.ndarray):
            image_array = np.array(image)
        else:
            image_array = image
        # assert (height, width, 3)
        image_array = image_array[:,:,::-1].astype(np.float32)
        image_array = torch.Tensor(image_array)
        if pixel_means is not None:
            if not isinstance(pixel_means, torch.Tensor):
                pixel_means = torch.Tensor(pixel_means)
            image_array -= pixel_means[None, None]
        image_array = image_array.permute(2, 0, 1)[None]
        # assert (1, 3, height, width)
    
        shape = image_array.shape
        size_min = min(shape[-2:])
        size_max = max(shape[-2:])
        scale = float(output_size_min_side) / float(size_min)
        if max_size and np.round(scale * size_max) > max_size:
            scale = float(max_size) / float(size_max)
        
        resized_image = torch.torch.nn.functional.interpolate(
            image_array, scale_factor=scale, mode='bilinear', align_corners=False
        )
        new_height, new_width = resized_image.shape[-2:]
        image_info = torch.Tensor([[new_height, new_width, scale]])
        return resized_image, image_info
    
    def detect(self, image, conf_thresh=0.4, min_boxes=10, max_boxes=20):
        """
        Detect regions from an image
        arguments:
            image: PIL.Image or numpy.ndarray
            conf_thresh: float
            min_boxes: int
            max_boxes: int
        returns:
            list of DetectedRegion
        """
        
        nms_threshold = self.model_config.NMS
        scale = self.model_config.IMAGE_SCALES[0]
        max_size = self.model_config.MAX_SIZE
        
        images, image_infos = self.convert_image_to_tensor(image, scale, max_size, self.pixel_means)
        
        # ToDo: HAS_RPN
        # ToDo: DEDUP_BOXES (make invert id list)
    
        # axis 0 in inputs should be 1
        # because this model does not tell batch ids of detected boxes 
        with torch.no_grad():
            outputs = self(images, image_infos, output_prob=True)
            # A dict that contains {'rois', 'cls_prob', 'attr_prob'}
    
        object_label_probs = outputs['cls_prob'] # cfg.TEST.SVM ==TRUE -> score
        num_object_labels = object_label_probs.shape[1]
        attribute_label_probs = outputs['attr_prob']
   
        boxes = outputs['rois'][:, 1:] / image_infos[:, 2, None]
        # get coordinates in the raw image scale; exclude axis 0 (for batch id)
        
        if self.model_config.BBOX_REG:
            # Fine-tuning bbox for each object class
            num_boxes = boxes.shape[0]
            boxes = boxes[:, :, None].repeat(1, num_object_labels, 1).view(num_boxes, - 1)
            boxes = self.proposal.bbox_transform_inv(boxes, outputs['bbox_pred'])
            boxes = self.proposal.clip_boxes_to_image(boxes, tuple(image_infos[0, :2] - 1))
            boxes = boxes.view(num_boxes, num_object_labels, 4)
        else:
            # Simply repeat the boxes, once for each class
            # add a class axis
            boxes = boxes[:, None, :].repeat(1, num_object_labels, 1)
    
        # ToDo: DEDUP_BOXES (restore arrays, object_label_prob and pred_boxes)
    
        # Keep only the best detections
        max_conf = torch.zeros((boxes.shape[0],))
        for object_id in range(1, num_object_labels):
            object_label_score = object_label_probs[:, object_id]
            box = boxes[:, object_id]
            keep = self.proposal.nms(box, object_label_score, nms_threshold)
            # choose items whose keep flag is True and score is the best so far
            max_conf[keep] = torch.where(
                object_label_score[keep] > max_conf[keep], 
                object_label_score[keep], max_conf[keep]
            )
        keep_boxes = torch.where(max_conf >= conf_thresh)[0]
    
        # back to ndarray
        boxes = boxes.cpu().numpy() 
        max_conf = max_conf.cpu().numpy()
        keep_boxes = keep_boxes.cpu().numpy()
        object_label_probs = object_label_probs.cpu().numpy() # cfg.TEST.SVM ==TRUE -> score
        attribute_label_probs = attribute_label_probs.cpu().numpy()
    
        # Adjust expected format
        if len(keep_boxes) < min_boxes:
            keep_boxes = np.argsort(max_conf)[::-1][:min_boxes]
        elif len(keep_boxes) > max_boxes:
            keep_boxes = np.argsort(max_conf)[::-1][:max_boxes]
    
        # initialize DetectedRegion
        regions = []
        for box, object_label_prob, attribute_label_prob in zip(
            boxes[keep_boxes], 
            object_label_probs[keep_boxes], 
            attribute_label_probs[keep_boxes]
        ):
            object_label_id = np.argmax(object_label_prob[1:], axis=0) + 1
            attribute_label_id = np.argmax(attribute_label_prob[1:], axis=0) + 1
            regions.append(DetectedRegion(
                box = box[object_label_id],
                object_label_id = object_label_id,
                attribute_label_id = attribute_label_id,
                object_label_conf = object_label_prob[object_label_id],
                attribute_label_conf = attribute_label_prob[attribute_label_id],
                object_label = self.object_vocab.get(object_label_id),
                attribute_label = self.attribute_vocab.get(attribute_label_id),
            ))
        
        return regions
    
    def forward(self, x, im_info, output_prob=True):
        
        res4b22 = self._forward_s1_s4(x)
        res4b22_size = res4b22.shape[-2:]
        # scaled down by 1/16 = 0.0625
        
        rpn_output = self.rpn_conv_3x3(res4b22)
        rpn_output = torch.nn.functional.relu(rpn_output)
        rpn_cls_score = self.rpn_cls_score(rpn_output)
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_output)
        # (batch_size, 4*num_anchors, height, width)
        
        rpn_cls_score_reshape = torch.reshape(rpn_cls_score, shape=(1, 2, -1)+res4b22_size)
        # (batch_size, 2, num_anchors, height, width); 2 account for background or foreground
        rpn_cls_prob = torch.nn.functional.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob_reshape = torch.reshape(rpn_cls_prob, shape=rpn_cls_score.shape)
        # (batch_size, 2*num_anchors, height, width)
        
        rois, _ = self.proposal(rpn_cls_prob_reshape, rpn_bbox_pred, im_info)
        
        roipool5 = torchvision.ops.roi_pool(res4b22, rois, output_size=(14, 14), spatial_scale=0.0625)
        pool5 = self._forward_s5(roipool5)
        pool5_flat = torch.flatten(pool5, start_dim=1, end_dim=-1)
        
        cls_score = self.cls_score(pool5)
        cls_prob = torch.nn.functional.softmax(cls_score, dim=-1)
        predicted_cls = torch.argmax(cls_score, dim=1)
        
        # delta for a box for each class 
        bbox_pred = self.bbox_pred(pool5)
        
        cls_embedding = self.cls_embedding(predicted_cls)
        cls_embedding_flat = torch.flatten(cls_embedding, start_dim=1, end_dim=-1)
        concat_pool5 = torch.cat([pool5_flat, cls_embedding_flat], dim=1)
        fc_attr = self.fc_attr(concat_pool5)
        fc_attr = torch.nn.functional.relu(fc_attr)
        
        attr_score = self.attr_score(fc_attr)
        attr_prob = torch.nn.functional.softmax(attr_score, dim=-1)
        
        outputs = {
            'rois': rois,
            'bbox_pred': bbox_pred,
        }
        if output_prob:
            outputs['cls_prob'] = cls_prob
            outputs['attr_prob'] = attr_prob
        else:
            outputs['cls_score'] = cls_score
            outputs['attr_score'] = attr_score
        
        return outputs
    
    def _check_model_config(self, model_config):
        
        assert len(model_config.IMAGE_SCALES) == 1, \
            f'This model support just a single scale: {len(model_config.IMAGE_SCALES)}'
    
    def _local_vocab(self, file_path, init_list=None):
        
        vocab = {i:v for i, v in enumerate(init_list or [])}
        
        if file_path is None:
            return vocab
        
        with open(file_path) as f:
            for line in f.readlines():
                vocab[len(vocab)] = line.split(',')[0].lower().strip()
        return vocab
    
    def __init__(self, model_config=None):
        
        super().__init__()
        
        # Configuration
        if model_config is None:
            model_config = ModelConfig()
        self._check_model_config(model_config)
        self.model_config = model_config
        self.pixel_means = torch.Tensor(self.model_config.PIXEL_MEANS)
        
        self.object_vocab = self._local_vocab(
            self.model_config.objects_vocab_path, ['__background__']
        )
        self.attribute_vocab = self._local_vocab(
            self.model_config.attributes_vocab_path, ['__no_attribute__']
        )
        
        # Proposal Module
        self.proposal = ProposalModule(model_config)
        
        # Network
        self.conv1 = torch.nn.Conv2d(3, 64, (7, 7), stride=(2, 2), padding=(3, 3),  dilation=1, groups=1, bias=False)
        self.bn_conv1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        
        self.res2a_branch1 = torch.nn.Conv2d(64, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn2a_branch1 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res2a_branch2a = torch.nn.Conv2d(64, 64, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn2a_branch2a = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res2a_branch2b = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn2a_branch2b = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res2a_branch2c = torch.nn.Conv2d(64, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn2a_branch2c = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res2b_branch2a = torch.nn.Conv2d(256, 64, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn2b_branch2a = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res2b_branch2b = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn2b_branch2b = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res2b_branch2c = torch.nn.Conv2d(64, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn2b_branch2c = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res2c_branch2a = torch.nn.Conv2d(256, 64, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn2c_branch2a = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res2c_branch2b = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn2c_branch2b = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res2c_branch2c = torch.nn.Conv2d(64, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn2c_branch2c = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        
        self.res3a_branch1 = torch.nn.Conv2d(256, 512, (1, 1), stride=(2, 2), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn3a_branch1 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3a_branch2a = torch.nn.Conv2d(256, 128, (1, 1), stride=(2, 2), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn3a_branch2a = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3a_branch2b = torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn3a_branch2b = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3a_branch2c = torch.nn.Conv2d(128, 512, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn3a_branch2c = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3b1_branch2a = torch.nn.Conv2d(512, 128, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn3b1_branch2a = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3b1_branch2b = torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn3b1_branch2b = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3b1_branch2c = torch.nn.Conv2d(128, 512, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn3b1_branch2c = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3b2_branch2a = torch.nn.Conv2d(512, 128, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn3b2_branch2a = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3b2_branch2b = torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn3b2_branch2b = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3b2_branch2c = torch.nn.Conv2d(128, 512, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn3b2_branch2c = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3b3_branch2a = torch.nn.Conv2d(512, 128, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn3b3_branch2a = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3b3_branch2b = torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn3b3_branch2b = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res3b3_branch2c = torch.nn.Conv2d(128, 512, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn3b3_branch2c = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        
        self.res4a_branch1 = torch.nn.Conv2d(512, 1024, (1, 1), stride=(2, 2), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4a_branch1 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4a_branch2a = torch.nn.Conv2d(512, 256, (1, 1), stride=(2, 2), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4a_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4a_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4a_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4a_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4a_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b1_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b1_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b1_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b1_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b1_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b1_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b2_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b2_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b2_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b2_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b2_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b2_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b3_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b3_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b3_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b3_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b3_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b3_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b4_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b4_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b4_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b4_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b4_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b4_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b5_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b5_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b5_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b5_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b5_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b5_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b6_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b6_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b6_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b6_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b6_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b6_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b7_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b7_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b7_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b7_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b7_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b7_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b8_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b8_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b8_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b8_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b8_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b8_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b9_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b9_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b9_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b9_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b9_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b9_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b10_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b10_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b10_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b10_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b10_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b10_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b11_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b11_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b11_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b11_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b11_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b11_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b12_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b12_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b12_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b12_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b12_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b12_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b13_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b13_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b13_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b13_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b13_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b13_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b14_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b14_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b14_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b14_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b14_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b14_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b15_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b15_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b15_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b15_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b15_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b15_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b16_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b16_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b16_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b16_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b16_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b16_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b17_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b17_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b17_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b17_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b17_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b17_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b18_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b18_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b18_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b18_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b18_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b18_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b19_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b19_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b19_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b19_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b19_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b19_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b20_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b20_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b20_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b20_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b20_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b20_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b21_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b21_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b21_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b21_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b21_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b21_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b22_branch2a = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b22_branch2a = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b22_branch2b = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=False)
        self.bn4b22_branch2b = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res4b22_branch2c = torch.nn.Conv2d(256, 1024, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn4b22_branch2c = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        
        self.rpn_conv_3x3 = torch.nn.Conv2d(1024, 512, (3, 3), stride=(1, 1), padding=(1, 1),  dilation=1, groups=1, bias=True)
        self.rpn_cls_score = torch.nn.Conv2d(512, 24, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=True)
        self.rpn_bbox_pred = torch.nn.Conv2d(512, 48, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=True)
        
        self.res5a_branch1 = torch.nn.Conv2d(1024, 2048, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn5a_branch1 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res5a_branch2a = torch.nn.Conv2d(1024, 512, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn5a_branch2a = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res5a_branch2b = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(2, 2),  dilation=2, groups=1, bias=False)
        self.bn5a_branch2b = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res5a_branch2c = torch.nn.Conv2d(512, 2048, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn5a_branch2c = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res5b_branch2a = torch.nn.Conv2d(2048, 512, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn5b_branch2a = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res5b_branch2b = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(2, 2),  dilation=2, groups=1, bias=False)
        self.bn5b_branch2b = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res5b_branch2c = torch.nn.Conv2d(512, 2048, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn5b_branch2c = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res5c_branch2a = torch.nn.Conv2d(2048, 512, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn5c_branch2a = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res5c_branch2b = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(2, 2),  dilation=2, groups=1, bias=False)
        self.bn5c_branch2b = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.res5c_branch2c = torch.nn.Conv2d(512, 2048, (1, 1), stride=(1, 1), padding=(0, 0),  dilation=1, groups=1, bias=False)
        self.bn5c_branch2c = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.0010000000000000009, affine=True)
        self.pool5 = torch.nn.AvgPool2d(kernel_size=(14, 14), stride=(1, 1), padding=(0, 0))
        
        self.cls_score = torch.nn.Linear(2048, 1601, bias=True)
        self.bbox_pred = torch.nn.Linear(2048, 6404, bias=True)
        self.cls_embedding = torch.nn.Embedding(1601, 256, padding_idx=None)
        self.fc_attr = torch.nn.Linear(2304, 512, bias=True)
        self.attr_score = torch.nn.Linear(512, 401, bias=True)
    
    def _forward_s1_s4(self, x):
        
        conv1 = self.conv1(x)
        conv1 = self.bn_conv1(conv1)
        conv1 = torch.nn.functional.relu(conv1)

        pool1 = self.pool1(conv1)
        res2a_branch1 = self.res2a_branch1(pool1)
        res2a_branch1 = self.bn2a_branch1(res2a_branch1)
        res2a_branch2a = self.res2a_branch2a(pool1)
        res2a_branch2a = self.bn2a_branch2a(res2a_branch2a)
        res2a_branch2a = torch.nn.functional.relu(res2a_branch2a)

        res2a_branch2b = self.res2a_branch2b(res2a_branch2a)
        res2a_branch2b = self.bn2a_branch2b(res2a_branch2b)
        res2a_branch2b = torch.nn.functional.relu(res2a_branch2b)

        res2a_branch2c = self.res2a_branch2c(res2a_branch2b)
        res2a_branch2c = self.bn2a_branch2c(res2a_branch2c)
        res2a = res2a_branch1 + res2a_branch2c
        res2a = torch.nn.functional.relu(res2a)

        res2b_branch2a = self.res2b_branch2a(res2a)
        res2b_branch2a = self.bn2b_branch2a(res2b_branch2a)
        res2b_branch2a = torch.nn.functional.relu(res2b_branch2a)

        res2b_branch2b = self.res2b_branch2b(res2b_branch2a)
        res2b_branch2b = self.bn2b_branch2b(res2b_branch2b)
        res2b_branch2b = torch.nn.functional.relu(res2b_branch2b)

        res2b_branch2c = self.res2b_branch2c(res2b_branch2b)
        res2b_branch2c = self.bn2b_branch2c(res2b_branch2c)
        res2b = res2a + res2b_branch2c
        res2b = torch.nn.functional.relu(res2b)

        res2c_branch2a = self.res2c_branch2a(res2b)
        res2c_branch2a = self.bn2c_branch2a(res2c_branch2a)
        res2c_branch2a = torch.nn.functional.relu(res2c_branch2a)

        res2c_branch2b = self.res2c_branch2b(res2c_branch2a)
        res2c_branch2b = self.bn2c_branch2b(res2c_branch2b)
        res2c_branch2b = torch.nn.functional.relu(res2c_branch2b)

        res2c_branch2c = self.res2c_branch2c(res2c_branch2b)
        res2c_branch2c = self.bn2c_branch2c(res2c_branch2c)
        res2c = res2b + res2c_branch2c
        res2c = torch.nn.functional.relu(res2c)

        res3a_branch1 = self.res3a_branch1(res2c)
        res3a_branch1 = self.bn3a_branch1(res3a_branch1)
        res3a_branch2a = self.res3a_branch2a(res2c)
        res3a_branch2a = self.bn3a_branch2a(res3a_branch2a)
        res3a_branch2a = torch.nn.functional.relu(res3a_branch2a)

        res3a_branch2b = self.res3a_branch2b(res3a_branch2a)
        res3a_branch2b = self.bn3a_branch2b(res3a_branch2b)
        res3a_branch2b = torch.nn.functional.relu(res3a_branch2b)

        res3a_branch2c = self.res3a_branch2c(res3a_branch2b)
        res3a_branch2c = self.bn3a_branch2c(res3a_branch2c)
        res3a = res3a_branch1 + res3a_branch2c
        res3a = torch.nn.functional.relu(res3a)

        res3b1_branch2a = self.res3b1_branch2a(res3a)
        res3b1_branch2a = self.bn3b1_branch2a(res3b1_branch2a)
        res3b1_branch2a = torch.nn.functional.relu(res3b1_branch2a)

        res3b1_branch2b = self.res3b1_branch2b(res3b1_branch2a)
        res3b1_branch2b = self.bn3b1_branch2b(res3b1_branch2b)
        res3b1_branch2b = torch.nn.functional.relu(res3b1_branch2b)

        res3b1_branch2c = self.res3b1_branch2c(res3b1_branch2b)
        res3b1_branch2c = self.bn3b1_branch2c(res3b1_branch2c)
        res3b1 = res3a + res3b1_branch2c
        res3b1 = torch.nn.functional.relu(res3b1)

        res3b2_branch2a = self.res3b2_branch2a(res3b1)
        res3b2_branch2a = self.bn3b2_branch2a(res3b2_branch2a)
        res3b2_branch2a = torch.nn.functional.relu(res3b2_branch2a)

        res3b2_branch2b = self.res3b2_branch2b(res3b2_branch2a)
        res3b2_branch2b = self.bn3b2_branch2b(res3b2_branch2b)
        res3b2_branch2b = torch.nn.functional.relu(res3b2_branch2b)

        res3b2_branch2c = self.res3b2_branch2c(res3b2_branch2b)
        res3b2_branch2c = self.bn3b2_branch2c(res3b2_branch2c)
        res3b2 = res3b1 + res3b2_branch2c
        res3b2 = torch.nn.functional.relu(res3b2)

        res3b3_branch2a = self.res3b3_branch2a(res3b2)
        res3b3_branch2a = self.bn3b3_branch2a(res3b3_branch2a)
        res3b3_branch2a = torch.nn.functional.relu(res3b3_branch2a)

        res3b3_branch2b = self.res3b3_branch2b(res3b3_branch2a)
        res3b3_branch2b = self.bn3b3_branch2b(res3b3_branch2b)
        res3b3_branch2b = torch.nn.functional.relu(res3b3_branch2b)

        res3b3_branch2c = self.res3b3_branch2c(res3b3_branch2b)
        res3b3_branch2c = self.bn3b3_branch2c(res3b3_branch2c)
        res3b3 = res3b2 + res3b3_branch2c
        res3b3 = torch.nn.functional.relu(res3b3)

        res4a_branch1 = self.res4a_branch1(res3b3)
        res4a_branch1 = self.bn4a_branch1(res4a_branch1)
        res4a_branch2a = self.res4a_branch2a(res3b3)
        res4a_branch2a = self.bn4a_branch2a(res4a_branch2a)
        res4a_branch2a = torch.nn.functional.relu(res4a_branch2a)

        res4a_branch2b = self.res4a_branch2b(res4a_branch2a)
        res4a_branch2b = self.bn4a_branch2b(res4a_branch2b)
        res4a_branch2b = torch.nn.functional.relu(res4a_branch2b)

        res4a_branch2c = self.res4a_branch2c(res4a_branch2b)
        res4a_branch2c = self.bn4a_branch2c(res4a_branch2c)
        res4a = res4a_branch1 + res4a_branch2c
        res4a = torch.nn.functional.relu(res4a)

        res4b1_branch2a = self.res4b1_branch2a(res4a)
        res4b1_branch2a = self.bn4b1_branch2a(res4b1_branch2a)
        res4b1_branch2a = torch.nn.functional.relu(res4b1_branch2a)

        res4b1_branch2b = self.res4b1_branch2b(res4b1_branch2a)
        res4b1_branch2b = self.bn4b1_branch2b(res4b1_branch2b)
        res4b1_branch2b = torch.nn.functional.relu(res4b1_branch2b)

        res4b1_branch2c = self.res4b1_branch2c(res4b1_branch2b)
        res4b1_branch2c = self.bn4b1_branch2c(res4b1_branch2c)
        res4b1 = res4a + res4b1_branch2c
        res4b1 = torch.nn.functional.relu(res4b1)

        res4b2_branch2a = self.res4b2_branch2a(res4b1)
        res4b2_branch2a = self.bn4b2_branch2a(res4b2_branch2a)
        res4b2_branch2a = torch.nn.functional.relu(res4b2_branch2a)

        res4b2_branch2b = self.res4b2_branch2b(res4b2_branch2a)
        res4b2_branch2b = self.bn4b2_branch2b(res4b2_branch2b)
        res4b2_branch2b = torch.nn.functional.relu(res4b2_branch2b)

        res4b2_branch2c = self.res4b2_branch2c(res4b2_branch2b)
        res4b2_branch2c = self.bn4b2_branch2c(res4b2_branch2c)
        res4b2 = res4b1 + res4b2_branch2c
        res4b2 = torch.nn.functional.relu(res4b2)

        res4b3_branch2a = self.res4b3_branch2a(res4b2)
        res4b3_branch2a = self.bn4b3_branch2a(res4b3_branch2a)
        res4b3_branch2a = torch.nn.functional.relu(res4b3_branch2a)

        res4b3_branch2b = self.res4b3_branch2b(res4b3_branch2a)
        res4b3_branch2b = self.bn4b3_branch2b(res4b3_branch2b)
        res4b3_branch2b = torch.nn.functional.relu(res4b3_branch2b)

        res4b3_branch2c = self.res4b3_branch2c(res4b3_branch2b)
        res4b3_branch2c = self.bn4b3_branch2c(res4b3_branch2c)
        res4b3 = res4b2 + res4b3_branch2c
        res4b3 = torch.nn.functional.relu(res4b3)

        res4b4_branch2a = self.res4b4_branch2a(res4b3)
        res4b4_branch2a = self.bn4b4_branch2a(res4b4_branch2a)
        res4b4_branch2a = torch.nn.functional.relu(res4b4_branch2a)

        res4b4_branch2b = self.res4b4_branch2b(res4b4_branch2a)
        res4b4_branch2b = self.bn4b4_branch2b(res4b4_branch2b)
        res4b4_branch2b = torch.nn.functional.relu(res4b4_branch2b)

        res4b4_branch2c = self.res4b4_branch2c(res4b4_branch2b)
        res4b4_branch2c = self.bn4b4_branch2c(res4b4_branch2c)
        res4b4 = res4b3 + res4b4_branch2c
        res4b4 = torch.nn.functional.relu(res4b4)

        res4b5_branch2a = self.res4b5_branch2a(res4b4)
        res4b5_branch2a = self.bn4b5_branch2a(res4b5_branch2a)
        res4b5_branch2a = torch.nn.functional.relu(res4b5_branch2a)

        res4b5_branch2b = self.res4b5_branch2b(res4b5_branch2a)
        res4b5_branch2b = self.bn4b5_branch2b(res4b5_branch2b)
        res4b5_branch2b = torch.nn.functional.relu(res4b5_branch2b)

        res4b5_branch2c = self.res4b5_branch2c(res4b5_branch2b)
        res4b5_branch2c = self.bn4b5_branch2c(res4b5_branch2c)
        res4b5 = res4b4 + res4b5_branch2c
        res4b5 = torch.nn.functional.relu(res4b5)

        res4b6_branch2a = self.res4b6_branch2a(res4b5)
        res4b6_branch2a = self.bn4b6_branch2a(res4b6_branch2a)
        res4b6_branch2a = torch.nn.functional.relu(res4b6_branch2a)

        res4b6_branch2b = self.res4b6_branch2b(res4b6_branch2a)
        res4b6_branch2b = self.bn4b6_branch2b(res4b6_branch2b)
        res4b6_branch2b = torch.nn.functional.relu(res4b6_branch2b)

        res4b6_branch2c = self.res4b6_branch2c(res4b6_branch2b)
        res4b6_branch2c = self.bn4b6_branch2c(res4b6_branch2c)
        res4b6 = res4b5 + res4b6_branch2c
        res4b6 = torch.nn.functional.relu(res4b6)

        res4b7_branch2a = self.res4b7_branch2a(res4b6)
        res4b7_branch2a = self.bn4b7_branch2a(res4b7_branch2a)
        res4b7_branch2a = torch.nn.functional.relu(res4b7_branch2a)

        res4b7_branch2b = self.res4b7_branch2b(res4b7_branch2a)
        res4b7_branch2b = self.bn4b7_branch2b(res4b7_branch2b)
        res4b7_branch2b = torch.nn.functional.relu(res4b7_branch2b)

        res4b7_branch2c = self.res4b7_branch2c(res4b7_branch2b)
        res4b7_branch2c = self.bn4b7_branch2c(res4b7_branch2c)
        res4b7 = res4b6 + res4b7_branch2c
        res4b7 = torch.nn.functional.relu(res4b7)

        res4b8_branch2a = self.res4b8_branch2a(res4b7)
        res4b8_branch2a = self.bn4b8_branch2a(res4b8_branch2a)
        res4b8_branch2a = torch.nn.functional.relu(res4b8_branch2a)

        res4b8_branch2b = self.res4b8_branch2b(res4b8_branch2a)
        res4b8_branch2b = self.bn4b8_branch2b(res4b8_branch2b)
        res4b8_branch2b = torch.nn.functional.relu(res4b8_branch2b)

        res4b8_branch2c = self.res4b8_branch2c(res4b8_branch2b)
        res4b8_branch2c = self.bn4b8_branch2c(res4b8_branch2c)
        res4b8 = res4b7 + res4b8_branch2c
        res4b8 = torch.nn.functional.relu(res4b8)

        res4b9_branch2a = self.res4b9_branch2a(res4b8)
        res4b9_branch2a = self.bn4b9_branch2a(res4b9_branch2a)
        res4b9_branch2a = torch.nn.functional.relu(res4b9_branch2a)

        res4b9_branch2b = self.res4b9_branch2b(res4b9_branch2a)
        res4b9_branch2b = self.bn4b9_branch2b(res4b9_branch2b)
        res4b9_branch2b = torch.nn.functional.relu(res4b9_branch2b)

        res4b9_branch2c = self.res4b9_branch2c(res4b9_branch2b)
        res4b9_branch2c = self.bn4b9_branch2c(res4b9_branch2c)
        res4b9 = res4b8 + res4b9_branch2c
        res4b9 = torch.nn.functional.relu(res4b9)

        res4b10_branch2a = self.res4b10_branch2a(res4b9)
        res4b10_branch2a = self.bn4b10_branch2a(res4b10_branch2a)
        res4b10_branch2a = torch.nn.functional.relu(res4b10_branch2a)

        res4b10_branch2b = self.res4b10_branch2b(res4b10_branch2a)
        res4b10_branch2b = self.bn4b10_branch2b(res4b10_branch2b)
        res4b10_branch2b = torch.nn.functional.relu(res4b10_branch2b)

        res4b10_branch2c = self.res4b10_branch2c(res4b10_branch2b)
        res4b10_branch2c = self.bn4b10_branch2c(res4b10_branch2c)
        res4b10 = res4b9 + res4b10_branch2c
        res4b10 = torch.nn.functional.relu(res4b10)

        res4b11_branch2a = self.res4b11_branch2a(res4b10)
        res4b11_branch2a = self.bn4b11_branch2a(res4b11_branch2a)
        res4b11_branch2a = torch.nn.functional.relu(res4b11_branch2a)

        res4b11_branch2b = self.res4b11_branch2b(res4b11_branch2a)
        res4b11_branch2b = self.bn4b11_branch2b(res4b11_branch2b)
        res4b11_branch2b = torch.nn.functional.relu(res4b11_branch2b)

        res4b11_branch2c = self.res4b11_branch2c(res4b11_branch2b)
        res4b11_branch2c = self.bn4b11_branch2c(res4b11_branch2c)
        res4b11 = res4b10 + res4b11_branch2c
        res4b11 = torch.nn.functional.relu(res4b11)

        res4b12_branch2a = self.res4b12_branch2a(res4b11)
        res4b12_branch2a = self.bn4b12_branch2a(res4b12_branch2a)
        res4b12_branch2a = torch.nn.functional.relu(res4b12_branch2a)

        res4b12_branch2b = self.res4b12_branch2b(res4b12_branch2a)
        res4b12_branch2b = self.bn4b12_branch2b(res4b12_branch2b)
        res4b12_branch2b = torch.nn.functional.relu(res4b12_branch2b)

        res4b12_branch2c = self.res4b12_branch2c(res4b12_branch2b)
        res4b12_branch2c = self.bn4b12_branch2c(res4b12_branch2c)
        res4b12 = res4b11 + res4b12_branch2c
        res4b12 = torch.nn.functional.relu(res4b12)

        res4b13_branch2a = self.res4b13_branch2a(res4b12)
        res4b13_branch2a = self.bn4b13_branch2a(res4b13_branch2a)
        res4b13_branch2a = torch.nn.functional.relu(res4b13_branch2a)

        res4b13_branch2b = self.res4b13_branch2b(res4b13_branch2a)
        res4b13_branch2b = self.bn4b13_branch2b(res4b13_branch2b)
        res4b13_branch2b = torch.nn.functional.relu(res4b13_branch2b)

        res4b13_branch2c = self.res4b13_branch2c(res4b13_branch2b)
        res4b13_branch2c = self.bn4b13_branch2c(res4b13_branch2c)
        res4b13 = res4b12 + res4b13_branch2c
        res4b13 = torch.nn.functional.relu(res4b13)

        res4b14_branch2a = self.res4b14_branch2a(res4b13)
        res4b14_branch2a = self.bn4b14_branch2a(res4b14_branch2a)
        res4b14_branch2a = torch.nn.functional.relu(res4b14_branch2a)

        res4b14_branch2b = self.res4b14_branch2b(res4b14_branch2a)
        res4b14_branch2b = self.bn4b14_branch2b(res4b14_branch2b)
        res4b14_branch2b = torch.nn.functional.relu(res4b14_branch2b)

        res4b14_branch2c = self.res4b14_branch2c(res4b14_branch2b)
        res4b14_branch2c = self.bn4b14_branch2c(res4b14_branch2c)
        res4b14 = res4b13 + res4b14_branch2c
        res4b14 = torch.nn.functional.relu(res4b14)

        res4b15_branch2a = self.res4b15_branch2a(res4b14)
        res4b15_branch2a = self.bn4b15_branch2a(res4b15_branch2a)
        res4b15_branch2a = torch.nn.functional.relu(res4b15_branch2a)

        res4b15_branch2b = self.res4b15_branch2b(res4b15_branch2a)
        res4b15_branch2b = self.bn4b15_branch2b(res4b15_branch2b)
        res4b15_branch2b = torch.nn.functional.relu(res4b15_branch2b)

        res4b15_branch2c = self.res4b15_branch2c(res4b15_branch2b)
        res4b15_branch2c = self.bn4b15_branch2c(res4b15_branch2c)
        res4b15 = res4b14 + res4b15_branch2c
        res4b15 = torch.nn.functional.relu(res4b15)

        res4b16_branch2a = self.res4b16_branch2a(res4b15)
        res4b16_branch2a = self.bn4b16_branch2a(res4b16_branch2a)
        res4b16_branch2a = torch.nn.functional.relu(res4b16_branch2a)

        res4b16_branch2b = self.res4b16_branch2b(res4b16_branch2a)
        res4b16_branch2b = self.bn4b16_branch2b(res4b16_branch2b)
        res4b16_branch2b = torch.nn.functional.relu(res4b16_branch2b)

        res4b16_branch2c = self.res4b16_branch2c(res4b16_branch2b)
        res4b16_branch2c = self.bn4b16_branch2c(res4b16_branch2c)
        res4b16 = res4b15 + res4b16_branch2c
        res4b16 = torch.nn.functional.relu(res4b16)

        res4b17_branch2a = self.res4b17_branch2a(res4b16)
        res4b17_branch2a = self.bn4b17_branch2a(res4b17_branch2a)
        res4b17_branch2a = torch.nn.functional.relu(res4b17_branch2a)

        res4b17_branch2b = self.res4b17_branch2b(res4b17_branch2a)
        res4b17_branch2b = self.bn4b17_branch2b(res4b17_branch2b)
        res4b17_branch2b = torch.nn.functional.relu(res4b17_branch2b)

        res4b17_branch2c = self.res4b17_branch2c(res4b17_branch2b)
        res4b17_branch2c = self.bn4b17_branch2c(res4b17_branch2c)
        res4b17 = res4b16 + res4b17_branch2c
        res4b17 = torch.nn.functional.relu(res4b17)

        res4b18_branch2a = self.res4b18_branch2a(res4b17)
        res4b18_branch2a = self.bn4b18_branch2a(res4b18_branch2a)
        res4b18_branch2a = torch.nn.functional.relu(res4b18_branch2a)

        res4b18_branch2b = self.res4b18_branch2b(res4b18_branch2a)
        res4b18_branch2b = self.bn4b18_branch2b(res4b18_branch2b)
        res4b18_branch2b = torch.nn.functional.relu(res4b18_branch2b)

        res4b18_branch2c = self.res4b18_branch2c(res4b18_branch2b)
        res4b18_branch2c = self.bn4b18_branch2c(res4b18_branch2c)
        res4b18 = res4b17 + res4b18_branch2c
        res4b18 = torch.nn.functional.relu(res4b18)

        res4b19_branch2a = self.res4b19_branch2a(res4b18)
        res4b19_branch2a = self.bn4b19_branch2a(res4b19_branch2a)
        res4b19_branch2a = torch.nn.functional.relu(res4b19_branch2a)

        res4b19_branch2b = self.res4b19_branch2b(res4b19_branch2a)
        res4b19_branch2b = self.bn4b19_branch2b(res4b19_branch2b)
        res4b19_branch2b = torch.nn.functional.relu(res4b19_branch2b)

        res4b19_branch2c = self.res4b19_branch2c(res4b19_branch2b)
        res4b19_branch2c = self.bn4b19_branch2c(res4b19_branch2c)
        res4b19 = res4b18 + res4b19_branch2c
        res4b19 = torch.nn.functional.relu(res4b19)

        res4b20_branch2a = self.res4b20_branch2a(res4b19)
        res4b20_branch2a = self.bn4b20_branch2a(res4b20_branch2a)
        res4b20_branch2a = torch.nn.functional.relu(res4b20_branch2a)

        res4b20_branch2b = self.res4b20_branch2b(res4b20_branch2a)
        res4b20_branch2b = self.bn4b20_branch2b(res4b20_branch2b)
        res4b20_branch2b = torch.nn.functional.relu(res4b20_branch2b)

        res4b20_branch2c = self.res4b20_branch2c(res4b20_branch2b)
        res4b20_branch2c = self.bn4b20_branch2c(res4b20_branch2c)
        res4b20 = res4b19 + res4b20_branch2c
        res4b20 = torch.nn.functional.relu(res4b20)

        res4b21_branch2a = self.res4b21_branch2a(res4b20)
        res4b21_branch2a = self.bn4b21_branch2a(res4b21_branch2a)
        res4b21_branch2a = torch.nn.functional.relu(res4b21_branch2a)

        res4b21_branch2b = self.res4b21_branch2b(res4b21_branch2a)
        res4b21_branch2b = self.bn4b21_branch2b(res4b21_branch2b)
        res4b21_branch2b = torch.nn.functional.relu(res4b21_branch2b)

        res4b21_branch2c = self.res4b21_branch2c(res4b21_branch2b)
        res4b21_branch2c = self.bn4b21_branch2c(res4b21_branch2c)
        res4b21 = res4b20 + res4b21_branch2c
        res4b21 = torch.nn.functional.relu(res4b21)

        res4b22_branch2a = self.res4b22_branch2a(res4b21)
        res4b22_branch2a = self.bn4b22_branch2a(res4b22_branch2a)
        res4b22_branch2a = torch.nn.functional.relu(res4b22_branch2a)

        res4b22_branch2b = self.res4b22_branch2b(res4b22_branch2a)
        res4b22_branch2b = self.bn4b22_branch2b(res4b22_branch2b)
        res4b22_branch2b = torch.nn.functional.relu(res4b22_branch2b)

        res4b22_branch2c = self.res4b22_branch2c(res4b22_branch2b)
        res4b22_branch2c = self.bn4b22_branch2c(res4b22_branch2c)
        res4b22 = res4b21 + res4b22_branch2c
        res4b22 = torch.nn.functional.relu(res4b22)
        
        return res4b22
    
    def _forward_s5(self, roipool5):
        
        res5a_branch1 = self.res5a_branch1(roipool5)
        res5a_branch1 = self.bn5a_branch1(res5a_branch1)
        res5a_branch2a = self.res5a_branch2a(roipool5)
        res5a_branch2a = self.bn5a_branch2a(res5a_branch2a)
        res5a_branch2a = torch.nn.functional.relu(res5a_branch2a)

        res5a_branch2b = self.res5a_branch2b(res5a_branch2a)
        res5a_branch2b = self.bn5a_branch2b(res5a_branch2b)
        res5a_branch2b = torch.nn.functional.relu(res5a_branch2b)

        res5a_branch2c = self.res5a_branch2c(res5a_branch2b)
        res5a_branch2c = self.bn5a_branch2c(res5a_branch2c)
        res5a = res5a_branch1 + res5a_branch2c
        res5a = torch.nn.functional.relu(res5a)

        res5b_branch2a = self.res5b_branch2a(res5a)
        res5b_branch2a = self.bn5b_branch2a(res5b_branch2a)
        res5b_branch2a = torch.nn.functional.relu(res5b_branch2a)

        res5b_branch2b = self.res5b_branch2b(res5b_branch2a)
        res5b_branch2b = self.bn5b_branch2b(res5b_branch2b)
        res5b_branch2b = torch.nn.functional.relu(res5b_branch2b)

        res5b_branch2c = self.res5b_branch2c(res5b_branch2b)
        res5b_branch2c = self.bn5b_branch2c(res5b_branch2c)
        res5b = res5a + res5b_branch2c
        res5b = torch.nn.functional.relu(res5b)

        res5c_branch2a = self.res5c_branch2a(res5b)
        res5c_branch2a = self.bn5c_branch2a(res5c_branch2a)
        res5c_branch2a = torch.nn.functional.relu(res5c_branch2a)

        res5c_branch2b = self.res5c_branch2b(res5c_branch2a)
        res5c_branch2b = self.bn5c_branch2b(res5c_branch2b)
        res5c_branch2b = torch.nn.functional.relu(res5c_branch2b)

        res5c_branch2c = self.res5c_branch2c(res5c_branch2b)
        res5c_branch2c = self.bn5c_branch2c(res5c_branch2c)
        res5c = res5b + res5c_branch2c
        res5c = torch.nn.functional.relu(res5c)

        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.shape[:2])
        
        return pool5
