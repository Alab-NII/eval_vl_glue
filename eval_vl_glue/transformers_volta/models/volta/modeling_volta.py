# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch VOLTA model. """


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    #BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_volta import VoltaConfig
from .vision_loss import vl_pretraining_losses

import copy


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "volta-base-uncased"
_CONFIG_FOR_DOC = "VoltaConfig"
_TOKENIZER_FOR_DOC = "VoltaTokenizer"

VOLTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    #"bert-base-uncased",
]


# Not implemented
# def load_tf_weights_in_volta(model, config, tf_checkpoint_path):
# class VoltaOnlyNSPHead(nn.Module):
# class VoltaPreTrainingHeads(nn.Module):
# class VoltaForPreTrainingOutput(ModelOutput):
# class VoltaForPreTraining(VoltaPreTrainedModel):
# class VoltaLMHeadModel(VoltaPreTrainedModel):
# class VoltaForNextSentencePrediction(VoltaPreTrainedModel):
# class VoltaForMultipleChoice(VoltaPreTrainedModel):
# class VoltaForTokenClassification(VoltaPreTrainedModel):
# class VoltaForQuestionAnswering(VoltaPreTrainedModel):


def coordinate_embeddings(boxes, dim):
    """
    Coordinate embeddings of bounding boxes
    :param boxes: [BS, K, 4] ([x1, y1, x2, y2])
    :param dim: sin/cos embedding dimension
    :return: [BS, K, 4, 2 * dim]
    """

    batch_size, num_boxes, num_loc = boxes.shape

    # transform to (x_c, y_c, w, h) format
    pos = boxes.new_zeros((batch_size, num_boxes, 4))
    pos[:, :, 0] = (boxes[:, :, 0] + boxes[:, :, 2]) / 2 * 100
    pos[:, :, 1] = (boxes[:, :, 1] + boxes[:, :, 3]) / 2 * 100
    pos[:, :, 2] = (boxes[:, :, 2] - boxes[:, :, 0]) * 100
    pos[:, :, 3] = (boxes[:, :, 3] - boxes[:, :, 1]) * 100

    # sin/cos embedding
    dim_mat = 1000 ** (torch.arange(dim, dtype=boxes.dtype, device=boxes.device) / float(dim))
    sin_embedding = (pos.view((batch_size, num_boxes, 4, 1)) / dim_mat.view((1, 1, 1, -1))).sin()
    cos_embedding = (pos.view((batch_size, num_boxes, 4, 1)) / dim_mat.view((1, 1, 1, -1))).cos()

    return torch.cat((sin_embedding, cos_embedding), dim=-1)


class FusedLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


@dataclass
class BaseModelOutputVL(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state: 
        hidden_states: optional
        attentions: optional
    """

    last_hidden_state_v: Tuple[torch.FloatTensor] = None
    last_hidden_state_l: Tuple[torch.FloatTensor] = None
    hidden_states_v: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_l: Optional[Tuple[torch.FloatTensor]] = None
    attentions_v: Optional[Tuple[torch.FloatTensor]] = None
    attentions_l: Optional[Tuple[torch.FloatTensor]] = None

        
@dataclass
class BaseModelOutputVLWithPooling(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state: 
        hidden_states: optional
        attentions: optional
    """

    last_hidden_state_v: Tuple[torch.FloatTensor] = None
    last_hidden_state_l: Tuple[torch.FloatTensor] = None
    pooler_output_v: Tuple[torch.FloatTensor] = None
    pooler_output_l: Tuple[torch.FloatTensor] = None
    hidden_states_v: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_l: Optional[Tuple[torch.FloatTensor]] = None
    attentions_v: Optional[Tuple[torch.FloatTensor]] = None
    attentions_l: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SequenceClassifierOutputVL(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss: optional
        logits:
        hidden_states_v: optional
        hidden_states_l: optional
        attentions_v: optional
        attentions_l: optional
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states_v: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_l: Optional[Tuple[torch.FloatTensor]] = None
    attentions_v: Optional[Tuple[torch.FloatTensor]] = None
    attentions_l: Optional[Tuple[torch.FloatTensor]] = None

        
class VoltaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VoltaImagePredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        if config.image_head_ln:
            self.LayerNorm = FusedLayerNorm(config.v_hidden_size, eps=1e-12)
        else:
            self.LayerNorm = lambda x: x

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VoltaLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = VoltaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
        

class VoltaOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = VoltaLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class VoltaImagePredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VoltaImagePredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict({
            key: nn.Linear(config.v_hidden_size, cls.input_dim)
            for key, cls in vl_pretraining_losses.items()
            if config.visual_target_weights.get(key, 0) > 0
        })

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for ix in self.decoder_dict:
            output[ix] = self.decoder_dict[ix](hidden_states)
        return output

    
class VoltaVLPreTrainingHeads(nn.Module):
    
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = VoltaLMPredictionHead(config, bert_model_embedding_weights)
        if config.fusion_method in {"none", "vl-bert_vqa"}:
            self.bi_seq_relationship = lambda x: None
        else:
            self.bi_seq_relationship = nn.Linear(config.pooler_size, 2)
        self.imagePredictions = VoltaImagePredictionHead(config)
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(0.1)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v):
        
        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        elif self.fusion_method == "text":
            pooled_output = self.dropout(pooled_output_t)
        elif self.fusion_method == "vl-bert_vqa":
            pooled_output = self.dropout(pooled_output_t)
        elif self.fusion_method == "none":
            pooled_output = None
        else:
            assert False

        prediction_scores_t = self.predictions(sequence_output_t)
        seq_relationship_score = self.bi_seq_relationship(pooled_output)
        prediction_scores_v_dict = self.imagePredictions(sequence_output_v)

        return prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, pooled_output


class SimpleClassifier(nn.Module):
    """
    Multi-layered Perceptron with GELU activation
    If hidden_dims is [], this is identical to a simple linear layer.
    """
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        
        layers = []
        last_dim = in_dim
        for d in hidden_dims:
            layers.append(nn.Linear(last_dim, d))
            layers.append(nn.GELU())
            layers.append(FusedLayerNorm(d, eps=1e-12))
            last_dim = d
        layers.append(nn.Linear(last_dim, out_dim))
        self.logit_fc = nn.Sequential(*layers)
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class VoltaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    
class VoltaEmbeddingsViLBertImage(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(config.num_locs, config.v_hidden_size)
        self.LayerNorm = FusedLayerNorm(config.v_hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)

        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class VoltaEmbeddingsLxmertImage(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(config.num_locs, config.v_hidden_size)
        self.ImgLayerNorm = FusedLayerNorm(config.v_hidden_size, eps=config.layer_norm_eps)
        self.LocLayerNorm = FusedLayerNorm(config.v_hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)

        img_embeddings = self.ImgLayerNorm(img_embeddings)
        loc_embeddings = self.LocLayerNorm(loc_embeddings)

        embeddings = (img_embeddings + loc_embeddings) / 2
        embeddings = self.dropout(embeddings)

        return embeddings


class VoltaEmbeddingsVLBert(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.with_mvrc_loss = config.visual_target_weights.get("6", 0) > 0
        self.initializer_range = config.initializer_range
        
        self.v_coordinate_embeddings_dim = config.v_coordinate_embeddings_dim
        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(config.v_attention_probs_dropout_prob),
            torch.nn.Linear(2 * config.v_feature_size, config.v_hidden_size),
            torch.nn.ReLU(inplace=True),
        )

        self.object_linguistic_embeddings = nn.Embedding(1, config.hidden_size)
        if self.with_mvrc_loss:
            self.object_mask_word_embedding = nn.Embedding(1, config.hidden_size)
        self.object_mask_visual_embedding = nn.Embedding(1, config.v_feature_size)
        self.end_embedding = nn.Embedding(1, config.hidden_size)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # visual transform
        self.visual_1x1_text = None
        self.visual_1x1_object = None
        if config.v_hidden_size != config.hidden_size:
            self.visual_1x1_text = nn.Linear(config.v_hidden_size, config.hidden_size)
            self.visual_1x1_object = nn.Linear(config.v_hidden_size, config.hidden_size)
        self.visual_ln_text = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_ln_object = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # init weights
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.obj_downsample[1].weight)
        self.object_mask_visual_embedding.weight.data.fill_(0.0)
        self.object_linguistic_embeddings.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if self.with_mvrc_loss:
            self.object_mask_word_embedding.weight.data.normal_(mean=0.0, std=self.initializer_range)
        self.visual_ln_text.weight.data.fill_(0.0)
        self.visual_ln_object.weight.data.fill_(0.0)
        self.LayerNorm.bias.data.zero_()
        self.LayerNorm.weight.data.fill_(1.0)

    def forward(self, token_ids, image_feat, image_loc, token_type_ids=None, position_ids=None):
        batch_size, num_boxes, _ = image_feat.shape

        mvrc_mask = (image_feat[:, :] == image_feat.new_zeros(image_feat.shape[-1])).sum(-1) == image_feat.shape[-1]
        image_feat[mvrc_mask] = self.object_mask_visual_embedding.weight[0]

        # Geometry Embedding + Appearance Feature
        coord_embed = coordinate_embeddings(image_loc, self.v_coordinate_embeddings_dim)
        feats_to_downsample = torch.cat(
            (coord_embed.view((batch_size * num_boxes, -1)), image_feat.view((batch_size * num_boxes, -1))),
            -1
        )
        final_feats = self.obj_downsample(feats_to_downsample).view(batch_size, num_boxes, -1)  # [BS, K, v_hidden]

        # Token Embedding for vision
        object_visual_embeddings = final_feats
        if self.visual_1x1_object is not None:
            object_visual_embeddings = self.visual_1x1_object(final_feats)
        object_visual_embeddings = self.visual_ln_object(object_visual_embeddings)  # [BS, K, v_hidden]
        object_linguistic_embeddings = self.object_linguistic_embeddings(
            final_feats.new_zeros((batch_size, num_boxes)).long()
        )  # [BS, K, v_hidden]
        if self.with_mvrc_loss:
            object_linguistic_embeddings[mvrc_mask] = self.object_mask_word_embedding.weight[0]
        _zero_id = torch.zeros((batch_size,), dtype=torch.long, device=object_linguistic_embeddings.device)
        object_linguistic_embeddings[:, -1] = self.end_embedding(_zero_id)
        object_vl_embeddings = object_linguistic_embeddings + object_visual_embeddings

        # Token Embedding + Visual Feature Embedding for text
        seq_length = token_ids.size(1)
        text_linguistic_embedding = self.word_embeddings(token_ids)
        # ToDo: is this right?
        text_visual_embeddings = final_feats[:, -1].repeat(1, seq_length).view(batch_size, seq_length, -1)
        if self.visual_1x1_text is not None:
            text_visual_embeddings = self.visual_1x1_text(text_visual_embeddings)
        text_visual_embeddings = self.visual_ln_text(text_visual_embeddings)
        text_vl_embeddings = text_linguistic_embedding + text_visual_embeddings

        # concatenate text and image
        text_mask = token_ids != 0
        text_end = text_mask.sum(1, keepdim=True)
        text_token_type_embeddings = self.token_type_embeddings(token_type_ids)
        object_type_ids = token_type_ids.new_zeros((batch_size, num_boxes)) + 2
        object_token_type_embeddings = self.token_type_embeddings(object_type_ids)

        # position embeddings
        text_position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        text_position_ids = text_position_ids.unsqueeze(0).expand_as(token_ids)
        text_position_ids[text_position_ids >= text_end] += num_boxes  # FIXME for variable number of objects
        object_position_ids = text_position_ids.new_zeros((batch_size, num_boxes))
        object_position_ids += text_end
        object_position_ids[:, -1] += 1
        text_position_embeddings = self.position_embeddings(text_position_ids)
        object_position_embeddings = self.position_embeddings(object_position_ids)

        embeddings = text_vl_embeddings + text_position_embeddings + text_token_type_embeddings
        v_embeddings = object_vl_embeddings + object_position_embeddings + object_token_type_embeddings
        vl_embeddings = torch.cat((embeddings, v_embeddings), dim=1)
        vl_embeddings = self.LayerNorm(vl_embeddings)
        vl_embeddings = self.dropout(vl_embeddings)
        embeddings, v_embeddings = vl_embeddings.split([embeddings.size(1), v_embeddings.size(1)], dim=1)

        return embeddings, v_embeddings

    
class VoltaEmbeddingsVisualBert(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Segment and position embedding for image features
        self.projection = nn.Linear(config.v_feature_size, config.hidden_size)
        self.token_type_embeddings_visual = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings_visual = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.special_initialize()

    def special_initialize(self):
        # This is a bit unorthodox. The better way might be to add an initializer to AllenNLP.
        # This function is used to init the token_type_embeddings_visual and position_embedding_visual, just in case.
        self.token_type_embeddings_visual.weight = torch.nn.Parameter(
            copy.deepcopy(self.token_type_embeddings.weight.data), requires_grad=True)
        self.position_embeddings_visual.weight = torch.nn.Parameter(
            copy.deepcopy(self.position_embeddings.weight.data), requires_grad=True)

    def forward(self, token_ids, image_feat, image_loc, token_type_ids=None, position_ids=None):
        batch_size, num_boxes, _ = image_feat.shape
        seq_length = token_ids.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(token_ids)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if image_feat is not None:
            visual_embeddings = self.projection(image_feat)
            visual_embeddings_type = token_type_ids.new_zeros((batch_size, num_boxes)) + 1
            token_type_embeddings_visual = self.token_type_embeddings_visual(visual_embeddings_type)

            image_text_alignment = None  # FIXME for VCR
            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the image feature. -1 is padding value.
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # position_embeddings_visual = Batch x image_length x alignment length x dim
                position_embeddings_visual = self.position_embeddings(image_text_alignment) * \
                    image_text_alignment_mask.to(dtype=next(self.parameters()).dtype).unsqueeze(-1)
                position_embeddings_visual = position_embeddings_visual.sum(2)

                # We want to average along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=next(self.parameters()).dtype).sum(2)
                image_text_alignment_mask[image_text_alignment_mask == 0] = 1  # Avoid divide by zero error
                position_embeddings_visual = position_embeddings_visual / image_text_alignment_mask.unsqueeze(-1)

                position_ids_visual = torch.zeros(*visual_embeddings.size()[:-1], dtype=torch.long).cuda()

                # When fine-tuning the detector, the image_text_alignment is sometimes padded too long.
                if position_embeddings_visual.size(1) != visual_embeddings.size(1):
                    assert(position_embeddings_visual.size(1) >= visual_embeddings.size(1))
                    position_embeddings_visual = position_embeddings_visual[:, :visual_embeddings.size(1), :]

                position_embeddings_visual = position_embeddings_visual + \
                    self.position_embeddings_visual(position_ids_visual)
            else:
                position_ids_visual = torch.zeros(*visual_embeddings.size()[:-1], dtype=torch.long).cuda()
                position_embeddings_visual = self.position_embeddings_visual(position_ids_visual)

            v_embeddings = visual_embeddings + position_embeddings_visual + token_type_embeddings_visual

            # Concat the two:
            vl_embeddings = torch.cat((embeddings, v_embeddings), dim=1)  # concat visual embeddings after attentions
            vl_embeddings = self.LayerNorm(vl_embeddings)
            vl_embeddings = self.dropout(vl_embeddings)
            embeddings, v_embeddings = vl_embeddings.split([embeddings.size(1), v_embeddings.size(1)], dim=1)
        else:
            v_embeddings = None
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)

        return embeddings, v_embeddings


class VoltaEmbeddingsUniter(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(config.num_locs, config.v_hidden_size)
        self.image_layer_norm = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.image_location_layer_norm = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.v_LayerNorm = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.v_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.special_initialize()

    def special_initialize(self):
        # This function is used to init v_LayerNorm as LayerNorm
        self.v_LayerNorm.weight = torch.nn.Parameter(copy.deepcopy(self.LayerNorm.weight.data), requires_grad=True)
        self.v_LayerNorm.bias = torch.nn.Parameter(copy.deepcopy(self.LayerNorm.bias.data), requires_grad=True)

    def forward(self, token_ids, image_feat, image_loc, token_type_ids=None, position_ids=None):
        batch_size, num_boxes, _ = image_feat.shape
        seq_length = token_ids.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(token_ids)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        img_embeddings = self.image_layer_norm(self.image_embeddings(image_feat))
        loc_embeddings = self.image_location_layer_norm(self.image_location_embeddings(image_loc))
        img_type_ids = torch.ones_like(image_feat[:, :, 0].long())
        v_token_type_embeddings = self.token_type_embeddings(img_type_ids)
        v_embeddings = img_embeddings + loc_embeddings + v_token_type_embeddings
        v_embeddings = self.v_LayerNorm(v_embeddings)
        v_embeddings = self.v_dropout(v_embeddings)

        return embeddings, v_embeddings

DUAL_EMBEDDINGS = {
    "vilbert": VoltaEmbeddingsViLBertImage,
    "lxmert": VoltaEmbeddingsLxmertImage,
}

SHARED_EMBEDDINGS = {
    "vl-bert": VoltaEmbeddingsVLBert,
    "visualbert": VoltaEmbeddingsVisualBert,
    "uniter": VoltaEmbeddingsUniter,
}


# Visualization vs output_attentions
class VoltaGatedSelfAttention(nn.Module):
    
    def __init__(self, config, layer_num):
        super().__init__()

        hidden_size = config.sublayer2attn_hidden_size.get(str(layer_num), config.hidden_size)
        num_attention_heads = config.sublayer2num_attention_heads.get(str(layer_num), config.num_attention_heads)
        v_hidden_size = config.sublayer2v_attn_hidden_size.get(str(layer_num), config.v_hidden_size)
        v_num_attention_heads = config.sublayer2v_num_attention_heads.get(str(layer_num), config.v_num_attention_heads)

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The text hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if v_hidden_size % v_num_attention_heads != 0:
            raise ValueError(
                "The vision hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (v_hidden_size, num_attention_heads)
            )
        self.v_num_attention_heads = v_num_attention_heads
        self.v_attention_head_size = int(v_hidden_size / v_num_attention_heads)
        self.v_all_head_size = self.v_num_attention_heads * self.v_attention_head_size

        self.visualization = config.visualization

        self.has_tt = (layer_num in config.tt_attn_sublayers)
        self.has_tv = (layer_num in config.tv_attn_sublayers)
        self.has_vt = (layer_num in config.vt_attn_sublayers)
        self.has_vv = (layer_num in config.vv_attn_sublayers)
        self.has_text = (self.has_tt or self.has_tv)
        self.has_vision = (self.has_vv or self.has_vt)
        self.share_layer = (layer_num in config.shared_sublayers)
        if self.has_tv or self.has_vt:
            assert hidden_size == v_hidden_size, "hidden_size != v_hidden_size"
            assert num_attention_heads == v_num_attention_heads, "num_attention_heads != v_num_attention_heads"

        if self.has_text:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        if self.has_text and self.has_vision and self.share_layer:
            assert hidden_size == v_hidden_size, "hidden_size != v_hidden_size"
            self.v_query = self.query
            self.v_key = self.key
            self.v_value = self.value
            self.v_dropout = self.dropout
        elif self.has_vision:
            self.v_query = nn.Linear(config.v_hidden_size, self.v_all_head_size)
            self.v_key = nn.Linear(config.v_hidden_size, self.v_all_head_size)
            self.v_value = nn.Linear(config.v_hidden_size, self.v_all_head_size)
            self.v_dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x, modality='text'):
        if modality == 'text':
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        else:
            new_x_shape = x.size()[:-1] + (self.v_num_attention_heads, self.v_attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, t_hidden_states, v_hidden_states, t_attention_mask, v_attention_mask):
        """
        Args:
            t_hidden_states: [bs, seq_len, hidden_size]
            v_hidden_states: [bs, num_box, v_hidden_size]
            t_attention_mask: [bs, 1, 1, seq_len] filled with 0s (non-pad) and -10000s (pad)
            v_attention_mask: [bs, 1, 1, num_box] filled with 0s (non-pad) and -10000s (pad)
        Returns:
            t_context_layer: [bs, seq_len, hidden_size] or int 0 if no text
            v_context_layer: [bs, num_box, v_hidden_size] or int 0 if no vision
            t_attn_data: dict or None if no visualization
            v_attn_data: dict or None if no visualization
        """
        if self.has_text:
            t_mixed_query_layer = self.query(t_hidden_states)  # [bs, seq_len, hidden_size]
            t_mixed_key_layer = self.key(t_hidden_states)
            t_mixed_value_layer = self.value(t_hidden_states)
            t_query_layer = self.transpose_for_scores(t_mixed_query_layer)  # [bs, num_heads, seq_len, attn_head_size]
            t_key_layer = self.transpose_for_scores(t_mixed_key_layer)
            t_value_layer = self.transpose_for_scores(t_mixed_value_layer)

        if self.has_vision:
            v_mixed_query_layer = self.v_query(v_hidden_states)  # [bs, num_box, v_hidden_size]
            v_mixed_key_layer = self.v_key(v_hidden_states)
            v_mixed_value_layer = self.v_value(v_hidden_states)
            v_query_layer = self.transpose_for_scores(v_mixed_query_layer, 'vision')  # [bs, v_num_heads, num_box, v_attn_head_size]
            v_key_layer = self.transpose_for_scores(v_mixed_key_layer, 'vision')
            v_value_layer = self.transpose_for_scores(v_mixed_value_layer, 'vision')

        # Gated attention
        if self.has_tt:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            tt_attention_scores = torch.matmul(t_query_layer, t_key_layer.transpose(-1, -2))  # [bs, num_heads, seq_len, seq_len]
            tt_attention_scores = tt_attention_scores / math.sqrt(self.attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            tt_attention_scores = tt_attention_scores + t_attention_mask  # [bs, num_heads, seq_len, seq_len]
        if self.has_tv:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            tv_attention_scores = torch.matmul(t_query_layer, v_key_layer.transpose(-1, -2))  # [bs, num_heads, seq_len, num_box]
            tv_attention_scores = tv_attention_scores / math.sqrt(self.attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            tv_attention_scores = tv_attention_scores + v_attention_mask
        if self.has_vt:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            vt_attention_scores = torch.matmul(v_query_layer, t_key_layer.transpose(-1, -2))  # [bs, num_heads, num_box, seq_len]
            vt_attention_scores = vt_attention_scores / math.sqrt(self.v_attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            vt_attention_scores = vt_attention_scores + t_attention_mask
        if self.has_vv:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            vv_attention_scores = torch.matmul(v_query_layer, v_key_layer.transpose(-1, -2))  # [bs, num_heads, num_box, num_box]
            vv_attention_scores = vv_attention_scores / math.sqrt(self.v_attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            vv_attention_scores = vv_attention_scores + v_attention_mask

        # Gated softmax
        # Normalize the attention scores to probabilities.
        if self.has_tt and self.has_tv:
            # Concatenate the two attention scores
            t_attention_scores = torch.cat((tt_attention_scores, tv_attention_scores), dim=-1)  # [bs, num_heads, seq_len, seq_len+num_box]
            t_attention_probs = nn.Softmax(dim=-1)(t_attention_scores)
            # Split concatenation back into tt and tv
            tt_attention_probs, tv_attention_probs = \
                t_attention_probs.split([tt_attention_scores.size(-1), tv_attention_scores.size(-1)], dim=-1)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            tt_attention_probs = self.dropout(tt_attention_probs)  # [bs, num_heads, seq_len, seq_len]
            tv_attention_probs = self.dropout(tv_attention_probs)  # [bs, num_heads, seq_len, num_box]
        elif self.has_tt:
            tt_attention_probs = self.dropout(nn.Softmax(dim=-1)(tt_attention_scores))  # [bs, num_heads, seq_len, seq_len]
        elif self.has_tv:
            tv_attention_probs = self.dropout(nn.Softmax(dim=-1)(tv_attention_scores))  # [bs, num_heads, seq_len, num_box]
        if self.has_vv and self.has_vt:
            # Concatenate the two attention scores
            v_attention_scores = torch.cat((vt_attention_scores, vv_attention_scores), dim=-1)  # [bs, num_heads, seq_len, seq_len+num_box]
            v_attention_probs = nn.Softmax(dim=-1)(v_attention_scores)
            # Split concatenation back into vt and vv
            vt_attention_probs, vv_attention_probs = \
                v_attention_probs.split([vt_attention_scores.size(-1), vv_attention_scores.size(-1)], dim=-1)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            vv_attention_probs = self.v_dropout(vv_attention_probs)  # [bs, num_heads, num_box, num_box]
            vt_attention_probs = self.v_dropout(vt_attention_probs)  # [bs, num_heads, num_box, seq_len]
        elif self.has_vv:
            vv_attention_probs = self.v_dropout(nn.Softmax(dim=-1)(vv_attention_scores))  # [bs, v_num_heads, num_box, num_box]
        elif self.has_vt:
            vt_attention_probs = self.v_dropout(nn.Softmax(dim=-1)(vt_attention_scores))  # [bs, num_heads, num_box, seq_len]

        # Gated context
        tt_context_layer, tv_context_layer, vt_context_layer, vv_context_layer = 0, 0, 0, 0
        if self.has_tt:
            tt_context_layer = torch.matmul(tt_attention_probs, t_value_layer)  # [bs, num_heads, seq_len, attn_head_size]
            tt_context_layer = tt_context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seq_len, num_heads, attn_head_size]
            tt_new_context_layer_shape = tt_context_layer.size()[:-2] + (self.all_head_size,)
            tt_context_layer = tt_context_layer.view(*tt_new_context_layer_shape)  # [bs, seq_len, num_heads*attn_head_size]
        if self.has_tv:
            tv_context_layer = torch.matmul(tv_attention_probs, v_value_layer)
            tv_context_layer = tv_context_layer.permute(0, 2, 1, 3).contiguous()
            tv_new_context_layer_shape = tv_context_layer.size()[:-2] + (self.all_head_size,)
            tv_context_layer = tv_context_layer.view(*tv_new_context_layer_shape)
        if self.has_vt:
            vt_context_layer = torch.matmul(vt_attention_probs, t_value_layer)
            vt_context_layer = vt_context_layer.permute(0, 2, 1, 3).contiguous()
            vt_new_context_layer_shape = vt_context_layer.size()[:-2] + (self.v_all_head_size,)
            vt_context_layer = vt_context_layer.view(*vt_new_context_layer_shape)
        if self.has_vv:
            vv_context_layer = torch.matmul(vv_attention_probs, v_value_layer)  # [bs, v_num_heads, num_box, v_attn_head_size]
            vv_context_layer = vv_context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, num_box, v_num_heads, v_attn_head_size]
            vv_new_context_layer_shape = vv_context_layer.size()[:-2] + (self.v_all_head_size,)
            vv_context_layer = vv_context_layer.view(*vv_new_context_layer_shape)  # [bs, num_box, v_num_heads*v_attn_head_size]

        t_context_layer = (tt_context_layer + tv_context_layer)  # [bs, seq_len, hidden_size] or int 0 if no text
        v_context_layer = (vv_context_layer + vt_context_layer)  # [bs, num_box, v_hidden_size] or int 0 if no vision

        if self.visualization:
            t_attn_data = {
                "intra_attn": tt_attention_probs if self.has_tt else None,
                "inter_attn": tv_attention_probs if self.has_tv else None,
                "queries": t_query_layer if self.has_text else None,
                "keys": t_key_layer if self.has_text else None,
            }
            v_attn_data = {
                "intra_attn": vv_attention_probs if self.has_vv else None,
                "inter_attn": vt_attention_probs if self.has_vt else None,
                "queries": v_query_layer if self.has_vision else None,
                "keys": v_key_layer if self.has_vision else None,
            }
        else:
            t_attn_data, v_attn_data = None, None

        return t_context_layer, v_context_layer, t_attn_data, v_attn_data


class VoltaGatedSelfOutput(nn.Module):
    
    def __init__(self, config, layer_num):
        super().__init__()
        
        hidden_size = config.sublayer2attn_hidden_size.get(str(layer_num), config.hidden_size)
        v_hidden_size = config.sublayer2v_attn_hidden_size.get(str(layer_num), config.v_hidden_size)

        self.has_language = ((layer_num in config.tt_attn_sublayers) or (layer_num in config.tv_attn_sublayers))
        self.has_vision = ((layer_num in config.vv_attn_sublayers) or (layer_num in config.vt_attn_sublayers))
        self.share_layer = (layer_num in config.shared_sublayers)
        self.single_ln = (layer_num in config.single_ln_sublayers)
        if self.single_ln:
            assert (self.has_language and self.has_vision and self.share_layer), "Missing language, vision or sharing"

        if self.has_language:
            self.dense = nn.Linear(hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.dense = lambda x: x
            self.dropout = lambda x: x
            self.LayerNorm = lambda x: x

        if self.has_language and self.has_vision and self.share_layer:
            assert (hidden_size == v_hidden_size) and (config.hidden_size == config.v_hidden_size), "hidden_size != v_hidden_size"
            self.v_dense = self.dense
            self.v_dropout = self.dropout
            self.v_LayerNorm = self.LayerNorm
        elif self.has_vision:
            self.v_dense = nn.Linear(v_hidden_size, config.v_hidden_size)
            self.v_dropout = nn.Dropout(config.v_hidden_dropout_prob)
            self.v_LayerNorm = FusedLayerNorm(v_hidden_size, eps=config.layer_norm_eps)
        else:
            self.v_dense = lambda x: x
            self.v_dropout = lambda x: x
            self.v_LayerNorm = lambda x: x

    def forward(self, t_hidden_states, v_hidden_states, t_input_tensor, v_input_tensor):
        """
        Args:
            t_hidden_states: [bs, seq_len, hidden_size] or int 0 if no text
            v_hidden_states: [bs, num_box, v_hidden_size] or int 0 if no vision
            t_input_tensor: [bs, seq_len, hidden_size]
            v_input_tensor: [bs, num_box, v_hidden_size]
        Returns:
            t_hidden_states: [bs, seq_len, hidden_size]
            v_hidden_states: [bs, num_box, v_hidden_size]
        """

        t_hidden_states = self.dense(t_hidden_states)
        t_hidden_states = self.dropout(t_hidden_states)
        v_hidden_states = self.v_dense(v_hidden_states)
        v_hidden_states = self.v_dropout(v_hidden_states)
        if self.single_ln:
            # Concatenate text and vision
            hidden_states = torch.cat((t_hidden_states, v_hidden_states), dim=1)  # [bs, seq_len+num_box, hidden_size]
            inputs = torch.cat((t_input_tensor, v_input_tensor), dim=1)  # [bs, seq_len+num_box, hidden_size]
            hidden_states = self.LayerNorm(hidden_states + inputs)
            t_hidden_states, v_hidden_states = \
                hidden_states.split([t_hidden_states.size(1), v_hidden_states.size(1)], dim=1)
        else:
            t_hidden_states = self.LayerNorm(t_hidden_states + t_input_tensor)
            v_hidden_states = self.v_LayerNorm(v_hidden_states + v_input_tensor)
        return t_hidden_states, v_hidden_states


class VoltaGatedAttention(nn.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.attention_self = VoltaGatedSelfAttention(config, layer_num)
        self.attention_output = VoltaGatedSelfOutput(config, layer_num)
    
    # not implemented
    # def prune_heads(self, heads):
    
    def forward(self, t_input_tensor, v_input_tensor, t_attention_mask, v_attention_mask):
        """
        Args:
            t_input_tensor: [bs, seq_len, hidden_size]
            v_input_tensor: [bs, num_box, v_hidden_size]
            t_attention_mask: [bs, 1, 1, seq_len] filled with 0s (non-pad) and -10000s (pad)
            v_attention_mask: [bs, 1, 1, num_box] filled with 0s (non-pad) and -10000s (pad)
        Returns:
            t_attn_output: [bs, seq_len, hidden_size]
            v_attn_output: [bs, num_box, v_hidden_size]
            t_attn_probs: dict or None if no visualization
            v_attn_probs: dict or None if no visualization
        """
        t_self_output, v_self_output, t_attn_probs, v_attn_probs = self.attention_self(t_input_tensor, v_input_tensor,
                                                                                       t_attention_mask, v_attention_mask)
        t_attn_output, v_attn_output = self.attention_output(t_self_output, v_self_output, t_input_tensor, v_input_tensor)
        return t_attn_output, v_attn_output, t_attn_probs, v_attn_probs

    
class VoltaGatedIntermediate(nn.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.has_language = (layer_num in config.t_ff_sublayers)
        self.has_vision = (layer_num in config.v_ff_sublayers)
        self.share_layer = (layer_num in config.shared_sublayers)

        intermediate_size = config.sublayer2intermediate_size.get(str(layer_num), config.intermediate_size)
        v_intermediate_size = config.sublayer2v_intermediate_size.get(str(layer_num), config.v_intermediate_size)

        if self.has_language:
            self.dense = nn.Linear(config.hidden_size, intermediate_size)
            if isinstance(config.hidden_act, str):
                self.intermediate_act_fn = ACT2FN[config.hidden_act]
            else:
                self.intermediate_act_fn = config.hidden_act
        else:
            self.dense = lambda x: x
            self.intermediate_act_fn = lambda x: 0
        if self.has_language and self.has_vision and self.share_layer:
            assert config.hidden_size == config.v_hidden_size, "hidden_size != v_hidden_size"
            assert intermediate_size == v_intermediate_size, "intermediate_size != v_intermediate_size"
            self.v_dense = self.dense
            self.v_intermediate_act_fn = self.intermediate_act_fn
        elif self.has_vision:
            self.v_dense = nn.Linear(config.v_hidden_size, v_intermediate_size)
            if isinstance(config.hidden_act, str):
                self.v_intermediate_act_fn = ACT2FN[config.v_hidden_act]
            else:
                self.v_intermediate_act_fn = config.v_hidden_act
        else:
            self.v_dense = lambda x: x
            self.v_intermediate_act_fn = lambda x: 0

    def forward(self, t_hidden_states, v_hidden_states):
        """
        Args:
            t_hidden_states: [bs, seq_len, hidden_size]
            v_hidden_states: [bs, num_box, v_hidden_size]
        Returns:
            t_hidden_states: [bs, seq_len, hidden_size] or int 0 if no text
            v_hidden_states: [bs, num_box, v_hidden_size] or int 0 if no vision
        """
        t_hidden_states = self.dense(t_hidden_states)
        t_hidden_states = self.intermediate_act_fn(t_hidden_states)

        v_hidden_states = self.v_dense(v_hidden_states)
        v_hidden_states = self.v_intermediate_act_fn(v_hidden_states)

        return t_hidden_states, v_hidden_states


class VoltaGatedOutput(nn.Module):
    
    def __init__(self, config, layer_num):
        super().__init__()
        self.has_language = (layer_num in config.t_ff_sublayers)
        self.has_vision = (layer_num in config.v_ff_sublayers)
        self.share_layer = (layer_num in config.shared_sublayers)
        self.single_ln = (layer_num in config.single_ln_sublayers)
        if self.single_ln:
            assert (self.has_language and self.has_vision and self.share_layer), "Missing language, vision or sharing"

        intermediate_size = config.sublayer2intermediate_size.get(str(layer_num), config.intermediate_size)
        v_intermediate_size = config.sublayer2v_intermediate_size.get(str(layer_num), config.v_intermediate_size)

        if self.has_language:
            self.dense = nn.Linear(intermediate_size, config.hidden_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.dense = lambda x: x
            self.dropout = lambda x: x
            self.LayerNorm = lambda x: x

        if self.has_language and self.has_vision and self.share_layer:
            assert config.hidden_size == config.v_hidden_size, "hidden_size != v_hidden_size"
            assert intermediate_size == v_intermediate_size, "intermediate_size != v_intermediate_size"
            self.v_dense = self.dense
            self.v_dropout = self.dropout
            self.v_LayerNorm = self.LayerNorm
        elif self.has_vision:
            self.v_dense = nn.Linear(v_intermediate_size, config.v_hidden_size)
            self.v_dropout = nn.Dropout(config.v_hidden_dropout_prob)
            self.v_LayerNorm = FusedLayerNorm(config.v_hidden_size, eps=config.layer_norm_eps)
        else:
            self.v_dense = lambda x: x
            self.v_dropout = lambda x: x
            self.v_LayerNorm = lambda x: x

    def forward(self, t_hidden_states, v_hidden_states, t_input_tensor, v_input_tensor):
        """
        Args:
            t_hidden_states: [bs, seq_len, hidden_size] or int 0 if no text
            v_hidden_states: [bs, num_box, v_hidden_size] or int 0 if no vision
            t_input_tensor: [bs, seq_len, hidden_size]
            v_input_tensor: [bs, num_box, v_hidden_size]
        Returns:
            t_hidden_states: [bs, seq_len, hidden_size]
            v_hidden_states: [bs, num_box, v_hidden_size]
        """
        t_hidden_states = self.dense(t_hidden_states)
        t_hidden_states = self.dropout(t_hidden_states)
        v_hidden_states = self.v_dense(v_hidden_states)
        v_hidden_states = self.v_dropout(v_hidden_states)
        if self.single_ln:
            # Concatenate text and vision
            hidden_states = torch.cat((t_hidden_states, v_hidden_states), dim=1)  # [bs, seq_len+num_box, hidden_size]
            inputs = torch.cat((t_input_tensor, v_input_tensor), dim=1)  # [bs, seq_len+num_box, hidden_size]
            hidden_states = self.LayerNorm(hidden_states + inputs)
            t_hidden_states, v_hidden_states = \
                hidden_states.split([t_hidden_states.size(1), v_hidden_states.size(1)], dim=1)
        else:
            t_hidden_states = self.LayerNorm(t_hidden_states + t_input_tensor)
            v_hidden_states = self.v_LayerNorm(v_hidden_states + v_input_tensor)
        return t_hidden_states, v_hidden_states


class VoltaGatedFeedForward(nn.Module):
    
    def __init__(self, config, layer_num):
        super().__init__()
        self.intermediate = VoltaGatedIntermediate(config, layer_num)
        self.output = VoltaGatedOutput(config, layer_num)

    #def feed_forward_chunk(self, attention_output):
    
    def forward(self, t_input_tensor, v_input_tensor):
        """
        Args:
            t_input_tensor: [bs, seq_len, hidden_size]
            v_input_tensor: [bs, num_box, v_hidden_size]
            # t_attention_probs: dict or None if no visualization
            # v_attention_probs: dict or None if no visualization
        Returns:
            t_layer_output: [bs, seq_len, hidden_size]
            v_layer_output: [bs, num_box, v_hidden_size]
            # t_attention_probs: dict or None if no visualization
            # v_attention_probs: dict or None if no visualization
        """
        t_inter_output, v_inter_output = self.intermediate(t_input_tensor, v_input_tensor)
        t_layer_output, v_layer_output = self.output(t_inter_output, v_inter_output, t_input_tensor, v_input_tensor)
        return t_layer_output, v_layer_output

    
class VoltaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.
        self.config = config
        attn_sublayers = set(config.tt_attn_sublayers + config.tv_attn_sublayers +
                             config.vt_attn_sublayers + config.vv_attn_sublayers)
        ff_sublayers = set(config.t_ff_sublayers + config.v_ff_sublayers)
        depth = len(attn_sublayers) + len(ff_sublayers)
        num2layers = {}
        self.num2type = {}
        for n in attn_sublayers:
            num2layers[n] = VoltaGatedAttention(config, n)
            self.num2type[n] = "attn"
        for n in ff_sublayers:
            num2layers[n] = VoltaGatedFeedForward(config, n)
            self.num2type[n] = "ff"
        assert len(num2layers) == depth, "Overlapping attn-ff sublayer numbers"
        assert (min(num2layers) == 0) and (max(num2layers) == depth - 1), "Non contiguous sublayer numbers"
        
        self.layer = nn.ModuleList([copy.deepcopy(sublayer) for _, sublayer in sorted(num2layers.items())])

    def forward(
        self,
        txt_embedding, # [bs, seq_len, hidden_size]
        img_embedding, # [bs, num_box, v_hidden_size]
        txt_attention_mask, # [bs, 1, 1, seq_len] filled with 0s (non-pad) and -10000s (pad)
        img_attention_mask, # [bs, 1, 1, num_box] filled with 0s (non-pad) and -10000s (pad)
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    ):
        all_encoder_layers_t = () if output_hidden_states else None
        all_encoder_layers_v = () if output_hidden_states else None

        all_attentions_t = () if output_attentions else None
        all_attentions_v = () if output_attentions else None
        
        hidden_states_t = txt_embedding
        hidden_states_v = img_embedding
        
        for idx, layer in enumerate(self.layer):
            if output_hidden_states:
                all_encoder_layers_t = all_encoder_layers_t + (hidden_states_t,)
                all_encoder_layers_v = all_encoder_layers_v + (hidden_states_v,)
            
            layer_type = self.num2type[idx]
            if layer_type == "attn":
                hidden_states_t, hidden_states_v, attention_t, attention_v = \
                    layer(hidden_states_t, hidden_states_v, txt_attention_mask, img_attention_mask)
            else:
                hidden_states_t, hidden_states_v = layer(hidden_states_t, hidden_states_v)
                attention_t = attention_v = None
            
            if output_attentions:
                all_attentions_t = all_attentions_t + (attention_t,)
                all_attentions_v = all_attentions_v + (attention_v,)

        if output_hidden_states:
            all_encoder_layers_t = all_encoder_layers_t + (hidden_states_t,)
            all_encoder_layers_v = all_encoder_layers_v + (hidden_states_v,)
        
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states_v,
                    hidden_states_t,
                    all_encoder_layers_v,
                    all_encoder_layers_t,
                    all_attentions_v,
                    all_attentions_t,
                ]
                if v is not None
            )
        return BaseModelOutputVL(
            last_hidden_state_v=hidden_states_v,
            last_hidden_state_l=hidden_states_t,
            hidden_states_v=all_encoder_layers_v,
            hidden_states_l=all_encoder_layers_t,
            attentions_v=all_attentions_v,
            attentions_l=all_attentions_t,
        )


class VoltaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.pooler_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding 
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class VoltaPoolerVLBertText(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.pooler_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, text_end):
        _, grid_pos = torch.meshgrid(
                torch.arange(hidden_states.size(0), dtype=torch.long, device=hidden_states.device),
                torch.arange(hidden_states.size(1), dtype=torch.long, device=hidden_states.device)
        )
        mask_token_tensor = hidden_states[(grid_pos == text_end - 2)]
        pooled_output = self.dense(mask_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class VoltaPoolerImage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_pooler_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding 
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class VoltaPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = VoltaConfig
    load_tf_weights = None
    base_model_prefix = "volta"
    _keys_to_ignore_on_load_missing = []
    
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


VOLTA_START_DOCSTRING = r"""(not provided)"""
VOLTA_INPUTS_DOCSTRING = r"""(not provided)"""


@add_start_docstrings(
    "The bare Volta Model transformer outputting raw hidden-states without any specific head on top.",
    VOLTA_START_DOCSTRING,
)
class VoltaModel(VoltaPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        
        self._pad_token_id = config.pad_token_id
        self.config = config
        
        self.shared_embeddings = False

        # initialize word embedding
        if config.model == "bert":
            self.embeddings = VoltaEmbeddings(config)
        else:
            raise Exception('choose config.model from ["bert"]. not %s' % config.model)

        # initialize vision embedding
        if config.image_embeddings in DUAL_EMBEDDINGS:
            self.v_embeddings = DUAL_EMBEDDINGS[config.image_embeddings](config)
        else:
            self.v_embeddings = lambda x, y: None
        
        if config.image_embeddings in SHARED_EMBEDDINGS:
            self.embeddings = SHARED_EMBEDDINGS[config.image_embeddings](config)
            self.shared_embeddings = True
        
        # make default image inputs
        num_boxes = config.default_num_boxes
        if config.add_global_imgfeat:
            num_boxes += 1
        self.register_buffer('dummy_input_imgs', torch.zeros((1, num_boxes, config.v_feature_size)))
        self.register_buffer('dummy_image_loc', torch.zeros((1, num_boxes, config.num_locs)))
        
        # encoder
        self.encoder = VoltaEncoder(config)
        
        # pooling
        self.fusion_method = config.fusion_method
        self.add_pooling_layer = add_pooling_layer
        if not add_pooling_layer:
            self.t_pooler = None
            self.v_pooler = None
        else:
            if config.fusion_method == "none":
                self.t_pooler = lambda x: None
            elif config.fusion_method == "vl-bert_vqa":
                self.t_pooler = VoltaPoolerVLBert(config)
            else:
                self.t_pooler = VoltaPooler(config)
        
            if config.fusion_method in {"none", "text", "vl-bert_vqa"}:
                self.v_pooler = lambda x: None
            else:
                assert config.pooler_size == config.v_pooler_size, "pooler_size != v_pooler_size"
                self.v_pooler = VoltaPoolerImage(config)
        
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    #def _prune_heads(self, heads_to_prune):
    
    def set_default_image_feature(self, image_feature):
        """
        Set default image feature to an existing model.
        Arguments:
            model: a volta model
            default_image_feature: a VoltaImageFeature
        Returns
            the given model
        """
        
        target = self.dummy_input_imgs
        # None means batch axis
        x = image_feature.features.clone()[None]
        assert x.shape == target.data.shape
        target.data = x
        
        target = self.dummy_image_loc
        # None means batch axis
        x = image_feature.image_location.clone()[None]
        assert x.shape == target.data.shape
        target.data = x
    
        return self
    
    #@add_start_docstrings_to_model_forward(VOLTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_start_docstrings_to_model_forward(VOLTA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputVL,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self, 
        input_ids, 
        input_images=None,
        image_loc=None, 
        token_type_ids=None, 
        attention_mask=None,
        image_attention_mask=None, 
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # make dummy image if input_imgs is None
        if input_images is None:
            mb_size = input_ids.size(0)
            input_images = self.dummy_input_imgs.repeat((mb_size, 1, 1))
            image_loc = self.dummy_image_loc.repeat((mb_size, 1, 1))
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(input_images.size(0), input_images.size(1)).type_as(input_ids)
        
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        # extended_attention_mask2 = attention_mask.unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_image_attention_mask = extended_image_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
        
        if self.shared_embeddings:
            embedding_output, v_embedding_output = self.embeddings(input_ids, input_images, image_loc, token_type_ids)
        else:
            embedding_output = self.embeddings(input_ids, token_type_ids)
            v_embedding_output = self.v_embeddings(input_images, image_loc)
        
        encoder_outputs = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output_v = encoder_outputs[0]
        sequence_output_l = encoder_outputs[1]
        
        if not self.add_pooling_layer:
            pooled_output_v = None
            pooled_output_l = None
        else:
            pooled_output_v = self.v_pooler(sequence_output_v)
            
            if self.fusion_method == "vl-bert_vqa":
                text_mask = input_ids != self._pad_token_id
                text_end = text_mask.sum(1, keepdim=True)
                pooled_output_l = self.t_pooler(sequence_output_l, text_end)
            else:
                pooled_output_l = self.t_pooler(sequence_output_l)
        
        if not return_dict:
            return (sequence_output_v, sequence_output_l, pooled_output_v, pooled_output_l) + encoder_outputs[2:]

        return BaseModelOutputVLWithPooling(
            last_hidden_state_v=sequence_output_v,
            last_hidden_state_l=sequence_output_l,
            pooler_output_v=pooled_output_v,
            pooler_output_l=pooled_output_l,
            hidden_states_v=encoder_outputs.hidden_states_v,
            hidden_states_l=encoder_outputs.hidden_states_l,
            attentions_v=encoder_outputs.attentions_v,
            attentions_l=encoder_outputs.attentions_l,
        )

    
class VoltaForVLPreTraining(VoltaPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)

        self.volta = VoltaModel(config)
        self.cls = VoltaVLPreTrainingHeads(config, self.volta.embeddings.word_embeddings.weight)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.visual_target_weights = config.visual_target_weights
        print("model's visual targets are ", [ix for ix, w in config.visual_target_weights.items() if w > 0])

        self.add_global_imgfeat = int(config.add_global_imgfeat is not None)

        self.init_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.volta.embeddings.word_embeddings)

    def forward(
        self,
        input_ids=None,
        input_images=None,
        image_loc=None,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        # ===== labels =====
        masked_lm_labels=None,
        image_label=None,
        image_cls=None,
        obj_labels=None,
        obj_confs=None,
        attr_labels=None,
        attr_confs=None,
        image_attrs=None,
        next_sentence_label=None,
        # ===== labels =====
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert False, 'We have not modified forward function for VL Pretraining.'
        # in this model, we first embed the images.
        #  Match output shape
        encoded_layers_t, encoded_layers_v, pooled_output_t, pooled_output_v, all_attention_mask = self.volta(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
        )
        if output_all_encoded_layers:
            sequence_output_t = encoded_layers_t[-1]
            sequence_output_v = encoded_layers_v[-1]
        else:
            sequence_output_t = encoded_layers_t
            sequence_output_v = encoded_layers_v

        prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, pooled_output = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )

        # Vision loss
        img_loss = 0
        for ix, weight in self.visual_target_weights.items():
            if self.config.add_global_imgfeat == "last":
                prediction_scores_v = prediction_scores_v_dict[ix][:, :-1]
            else:
                prediction_scores_v = prediction_scores_v_dict[ix][:, self.add_global_imgfeat:]
            # ref pre_vis_criterions in losses
            img_loss += pre_vis_criterions[ix](prediction_scores_v, weight, image_label, image_cls, image_feat,
                                               obj_labels, obj_confs, attr_labels, attr_confs)

        masked_img_loss = img_loss > 0 if type(img_loss) == int else img_loss.cpu().item() > 0
        if masked_img_loss:
            img_loss = img_loss.unsqueeze(0)
        else:
            img_loss = torch.zeros(1).cuda()

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(
                prediction_scores_t.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            ).unsqueeze(0)
        else:
            masked_lm_loss = torch.zeros(1).cuda()

        if (seq_relationship_score is not None) and (next_sentence_label is not None):
            next_sentence_loss = self.loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            ).unsqueeze(0)
        else:
            next_sentence_loss = torch.zeros(1).cuda()
        
        # Change output shape
        if masked_img_loss or masked_lm_loss or next_sentence_loss:
            if output_all_encoded_layers:
                return masked_lm_loss, img_loss, next_sentence_loss, encoded_layers_t, encoded_layers_v
            return masked_lm_loss, img_loss, next_sentence_loss
        else:
            if output_all_encoded_layers:
                return prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, all_attention_mask, \
                       pooled_output, encoded_layers_t, encoded_layers_v
            return prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, all_attention_mask, pooled_output

    
@add_start_docstrings("""Volta Model with a `language modeling` head on top. """, VOLTA_START_DOCSTRING)
class VoltaForMaskedLM(VoltaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        
        assert config.num_images == 1, 'VoltaForMaskedLM not support multiple images'
        
        if config.is_decoder:
            raise RuntimeError('Volta currently support only encoder')
        
        self.volta = VoltaModel(config, add_pooling_layer=False)
        self.cls = VoltaOnlyMLMHead(config, self.volta.embeddings.word_embeddings.weight)

        self.init_weights()
    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.volta.embeddings.word_embeddings)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(VOLTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        input_images=None,
        image_loc=None,        
        attention_mask=None,
        token_type_ids=None,
        #position_ids=None,
        image_attention_mask=None,
        #head_mask=None,
        #inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.volta(
            input_ids,
            input_images=input_images,
            image_loc=image_loc,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            #position_ids=position_ids,
            image_attention_mask=image_attention_mask,
            #head_mask=head_mask,
            #inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        #sequence_output_v = outputs[0]
        sequence_output_l = outputs[1]
        
        prediction_scores = self.cls(sequence_output_l)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        # change object for vl
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states_l,
            attentions=outputs.attentions_l,
        )

    #def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
    #    input_shape = input_ids.shape
    #    effective_batch_size = input_shape[0]
    #
    #    #  add a dummy token
    #    assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
    #    attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
    #    dummy_token = torch.full(
    #        (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
    #    )
    #    input_ids = torch.cat([input_ids, dummy_token], dim=1)
    #
    #    return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings(
    """
    Volta Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    VOLTA_START_DOCSTRING,
)
class VoltaForSequenceClassification(VoltaPreTrainedModel):
    
    all_fusion_methods = ['sum', 'mul', 'text', 'vl-bert_vqa' 'none']
    
    def __init__(self, config):
        super().__init__(config)
        
        self.num_images = config.num_images
        self.num_labels = config.num_labels
        self.fusion_method = config.fusion_method
        assert self.fusion_method in self.all_fusion_methods, 'fusion_method should be in %s' % self.all_fusion_methods
        
        self.volta = VoltaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if config.classifier_dims:
            self.classifier = SimpleClassifier(
                self.num_images*config.pooler_size, 
                config.classifier_dims, 
                config.num_labels
            )
        else:
            self.classifier = nn.Linear(self.num_images*config.pooler_size, config.num_labels)

        self.init_weights()
    
    def _shape_mb(self, t, k, requires_repeat=False):
        """Change the tensor shape for multiple images.
        This module assumes that multiple images are input by concatenating their feature sequences.
        For example, if the sequence length is 36 for an image and num_images are 2, 
        the shape of input_images is expected to be (mb_size, 36*2, feature_dim).
        This function, with requires_repeat=False, changes the shape to (mb_size*2, 36, feature_dim) by increasing mb_size.
        And when requires_repeat is True, this function changes the shape by simply repeating the input tensor along the second axis.
        Requires_repeat is used for non-visual inputs that corresponds each multiple image.
        Arguments:
            t: input tensor,
            k: the number of images
            requires_repeat: if set True, the input tensor will repeats to match minibatch size.
        Returns:
            shaped tensor
        """
        if requires_repeat:
            t = t.repeat(*((1, k)+(1,)*(len(t.shape)-2)))
        
        src_shape = t.shape
        dest_shape = (src_shape[0]*k, src_shape[1]//k) + src_shape[2:]
        
        return t.view(*dest_shape)
    
    #@add_start_docstrings_to_model_forward(VOLTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_start_docstrings_to_model_forward(VOLTA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        input_images=None,
        image_loc=None,
        attention_mask=None,
        token_type_ids=None,
        #position_ids=None,
        image_attention_mask=None,
        #head_mask=None,
        #inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Reshape minibatch for multiple images
        if self.num_images > 1:
            # vision context
            input_images = self._shape_mb(input_images, self.num_images, requires_repeat=False)
            image_loc = self._shape_mb(image_loc, self.num_images, requires_repeat=False)
            if image_attention_mask is not None:
                image_attention_mask = self._shape_mb(image_attention_mask, self.num_images, requires_repeat=False)
            
            # language context
            input_ids = self._shape_mb(input_ids, self.num_images, requires_repeat=True)
            if attention_mask is not None:
                attention_mask = self._shape_mb(attention_mask, self.num_images, requires_repeat=True)
            if token_type_ids is not None:
                token_type_ids = self._shape_mb(token_type_ids, self.num_images, requires_repeat=True)
        
        outputs = self.volta(
            input_ids,
            input_images=input_images,
            image_loc=image_loc,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            #position_ids=position_ids,
            image_attention_mask=image_attention_mask,
            #head_mask=head_mask,
            #inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooler_output_v = outputs[2]
        pooler_output_l = outputs[3]
        
        if self.fusion_method == "sum":
            pooler_output = pooler_output_l + pooler_output_v
        elif self.fusion_method == "mul":
            pooler_output = pooler_output_l * pooler_output_v
        elif self.fusion_method == "text":
            pooler_output = pooler_output_l 
        elif self.fusion_method == "vl-bert_vqa":
            pooler_output = pooler_output_l 
        elif self.fusion_method == "none":
            pooler_output = None
        
        # Reunify divided information by reshape mb_size and dim axes.
        if self.num_images > 1:
            pooler_output = pooler_output.view(-1, pooler_output.size(1) * self.num_images)
        
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[4:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputVL(
            loss=loss,
            logits=logits,
            hidden_states_v=outputs.hidden_states_v,
            hidden_states_l=outputs.hidden_states_l,
            attentions_v=outputs.attentions_v,
            attentions_l=outputs.attentions_l,
        )
