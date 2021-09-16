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
""" BERT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

VOLTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    #"bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
}


class VoltaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BertModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a BERT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.

    Examples::

        >>> from transformers import BertModel, BertConfig

        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> configuration = BertConfig()

        >>> # Initializing a model from the bert-base-uncased style configuration
        >>> model = BertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "volta"

    def __init__(
        self,
        #
        # From BERT
        vocab_size=30522,
        hidden_size=768,
        #num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        #gradient_checkpointing=False,
        #position_embedding_type="absolute",
        #use_cache=True,
        #
        # From Volta
        pooler_size=768,
        num_locs=5,
        model="bert",
        image_embeddings="vilbert",
        add_global_imgfeat=None,
        v_coordinate_embeddings_dim=None,
        v_feature_size=2048,
        v_hidden_size=768,
        v_num_attention_heads=12,
        v_intermediate_size=3072,
        v_pooler_size=1024,
        v_attention_probs_dropout_prob=0.1,
        v_hidden_act="gelu",
        v_hidden_dropout_prob=0.1,
        v_initializer_range=0.2,
        visual_target_weights={"0": 1},
        fixed_layers=[],
        fusion_method="mul",
        objective=0,
        #clf_hidden_size=1536,
        image_head_ln=True,
        visualization=False,
        tt_attn_sublayers=[],
        tv_attn_sublayers=[],
        vt_attn_sublayers=[],
        vv_attn_sublayers=[],
        t_ff_sublayers=[],
        v_ff_sublayers=[],
        shared_sublayers=[],
        single_ln_sublayers=[],
        sublayer2attn_hidden_size={},
        sublayer2num_attention_heads={},
        sublayer2intermediate_size={},
        sublayer2v_attn_hidden_size={},
        sublayer2v_num_attention_heads={},
        sublayer2v_intermediate_size={},
        bert_layer2attn_sublayer={},
        bert_layer2ff_sublayer={},
        default_num_boxes=36,
        num_images=1,
        classifier_dims=[],
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        # From BERT
        self.layer_norm_eps = layer_norm_eps
        #self.gradient_checkpointing = gradient_checkpointing
        #self.position_embedding_type = position_embedding_type
        #self.use_cache = use_cache
        
        # From Volta
        # Text
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pooler_size = pooler_size
        # Vision
        self.num_locs = num_locs
        self.v_coordinate_embeddings_dim = v_coordinate_embeddings_dim
        self.add_global_imgfeat = add_global_imgfeat
        self.image_embeddings = image_embeddings
        self.v_feature_size = v_feature_size
        self.v_hidden_size = v_hidden_size
        self.v_num_attention_heads = v_num_attention_heads
        self.v_intermediate_size = v_intermediate_size
        self.v_attention_probs_dropout_prob = v_attention_probs_dropout_prob
        self.v_hidden_act = v_hidden_act
        self.v_hidden_dropout_prob = v_hidden_dropout_prob
        self.v_initializer_range = v_initializer_range
        self.v_pooler_size = v_pooler_size
        # Text-Vision
        self.tt_attn_sublayers = tt_attn_sublayers
        self.tv_attn_sublayers = tv_attn_sublayers
        self.vt_attn_sublayers = vt_attn_sublayers
        self.vv_attn_sublayers = vv_attn_sublayers
        self.t_ff_sublayers = t_ff_sublayers
        self.v_ff_sublayers = v_ff_sublayers
        self.shared_sublayers = shared_sublayers
        self.single_ln_sublayers = single_ln_sublayers
        self.sublayer2attn_hidden_size = sublayer2attn_hidden_size
        self.sublayer2num_attention_heads = sublayer2num_attention_heads
        self.sublayer2intermediate_size = sublayer2intermediate_size
        self.sublayer2v_attn_hidden_size = sublayer2v_attn_hidden_size
        self.sublayer2v_num_attention_heads = sublayer2v_num_attention_heads
        self.sublayer2v_intermediate_size = sublayer2v_intermediate_size
        self.bert_layer2attn_sublayer = bert_layer2attn_sublayer
        self.bert_layer2ff_sublayer = bert_layer2ff_sublayer
        self.image_head_ln = image_head_ln
        # Else
        self.visual_target_weights = visual_target_weights
        self.fixed_layers = fixed_layers
        self.model = model
        # Pre-training
        self.fusion_method = fusion_method
        self.objective = objective
        # Fine-tuning
        #self.clf_hidden_size = clf_hidden_size
        self.visualization = visualization
        self.default_num_boxes = default_num_boxes
        self.num_images = num_images
        self.classifier_dims = classifier_dims
