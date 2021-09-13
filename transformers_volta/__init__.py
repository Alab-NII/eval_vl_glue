# cofing: utf-8
# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# Modified by Taichi Iki 
# Parts not related for our experiments were omitted.

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "4.4.0.dev0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from . import dependency_versions_check
from .file_utils import (
    _BaseLazyModule,
    is_torch_available,
)
from .utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Base objects, independent of any specific backend
_import_structure = {
    "configuration_utils": ["PretrainedConfig"],
    "data": [
        "DataProcessor",
        "InputExample",
        "InputFeatures",
        "SingleSentenceClassificationProcessor",
        "SquadExample",
        "SquadFeatures",
        "SquadV1Processor",
        "SquadV2Processor",
        "glue_compute_metrics",
        "glue_convert_examples_to_features",
        "glue_output_modes",
        "glue_processors",
        "glue_tasks_num_labels",
        "squad_convert_examples_to_features",
        "xnli_compute_metrics",
        "xnli_output_modes",
        "xnli_processors",
        "xnli_tasks_num_labels",
    ],
    "file_utils": [
        "CONFIG_NAME",
        "MODEL_CARD_NAME",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "SPIECE_UNDERLINE",
        "TF2_WEIGHTS_NAME",
        "TF_WEIGHTS_NAME",
        "TRANSFORMERS_CACHE",
        "WEIGHTS_NAME",
        "TensorType",
        "add_end_docstrings",
        "add_start_docstrings",
        "cached_path",
        "is_apex_available",
        "is_datasets_available",
        "is_faiss_available",
        "is_flax_available",
        "is_psutil_available",
        "is_py3nvml_available",
        "is_sentencepiece_available",
        "is_sklearn_available",
        "is_tf_available",
        "is_tokenizers_available",
        "is_torch_available",
        "is_torch_tpu_available",
    ],
    "hf_argparser": ["HfArgumentParser"],
    "models": [],
    # Models
    "models.auto": [
        "ALL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CONFIG_MAPPING",
        "MODEL_NAMES_MAPPING",
        "TOKENIZER_MAPPING",
        "AutoConfig",
        "AutoTokenizer",
    ],
    # ToDo: BERT
    # ToDo:Volta
    "tokenization_utils": ["PreTrainedTokenizer"],
    "tokenization_utils_base": [
        "AddedToken",
        "BatchEncoding",
        "CharSpan",
        "PreTrainedTokenizerBase",
        "SpecialTokensMixin",
        "TokenSpan",
    ],
    "trainer_callback": [
        "DefaultFlowCallback",
        "EarlyStoppingCallback",
        "PrinterCallback",
        "ProgressCallback",
        "TrainerCallback",
        "TrainerControl",
        "TrainerState",
    ],
    "trainer_utils": ["EvalPrediction", "IntervalStrategy", "SchedulerType", "set_seed"],
    "training_args": ["TrainingArguments"],
    "utils": ["logging"],
}

# sentencepiece-backed objects
from .utils import dummy_sentencepiece_objects
_import_structure["utils.dummy_sentencepiece_objects"] = [
    name for name in dir(dummy_sentencepiece_objects) if not name.startswith("_")
]

# tokenziers-backed objects
from .utils import dummy_tokenizers_objects
_import_structure["utils.dummy_tokenizers_objects"] = [
    name for name in dir(dummy_tokenizers_objects) if not name.startswith("_")
 ]

# PyTorch-backed objects
if is_torch_available():
    
    _import_structure["data.data_collator"] = [
        "DataCollator",
        "DataCollatorForLanguageModeling",
        "DataCollatorForPermutationLanguageModeling",
        "DataCollatorForSeq2Seq",
        "DataCollatorForSOP",
        "DataCollatorForTokenClassification",
        "DataCollatorForWholeWordMask",
        "DataCollatorWithPadding",
        "default_data_collator",
    ]
    _import_structure["data.datasets"] = [
        "GlueDataset",
        "GlueDataTrainingArguments",
        "LineByLineTextDataset",
        "LineByLineWithRefDataset",
        "LineByLineWithSOPTextDataset",
        "SquadDataset",
        "SquadDataTrainingArguments",
        "TextDataset",
        "TextDatasetForNextSentencePrediction",
    ]
    _import_structure["generation_beam_search"] = ["BeamScorer", "BeamSearchScorer"]
    _import_structure["generation_logits_process"] = [
        "HammingDiversityLogitsProcessor",
        "LogitsProcessor",
        "LogitsProcessorList",
        "LogitsWarper",
        "MinLengthLogitsProcessor",
        "NoBadWordsLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PrefixConstrainedLogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
    ]
    _import_structure["generation_utils"] = ["top_k_top_p_filtering"]
    _import_structure["modeling_utils"] = ["Conv1D", "PreTrainedModel", "apply_chunking_to_forward", "prune_layer"]
    # PyTorch models structure
    _import_structure["models.auto"].extend(
        [
            "MODEL_FOR_CAUSAL_LM_MAPPING",
            "MODEL_FOR_MASKED_LM_MAPPING",
            "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "MODEL_FOR_PRETRAINING_MAPPING",
            "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "MODEL_MAPPING",
            "MODEL_WITH_LM_HEAD_MAPPING",
            "AutoModel",
            "AutoModelForCausalLM",
            "AutoModelForMaskedLM",
            "AutoModelForMultipleChoice",
            "AutoModelForNextSentencePrediction",
            "AutoModelForPreTraining",
            "AutoModelForQuestionAnswering",
            "AutoModelForSeq2SeqLM",
            "AutoModelForSequenceClassification",
            "AutoModelForTableQuestionAnswering",
            "AutoModelForTokenClassification",
            "AutoModelWithLMHead",
        ]
    )
    # ToDo: BERT
    # ToDo:Volta
    _import_structure["optimization"] = [
        "Adafactor",
        "AdamW",
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ]
    _import_structure["trainer"] = ["Trainer"]
    #_import_structure["trainer_pt_utils"] = ["torch_distributed_zero_first"]
else:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]

# TensorFlow-backed objects
from .utils import dummy_tf_objects
_import_structure["utils.dummy_tf_objects"] = [name for name in dir(dummy_tf_objects) if not name.startswith("_")]

# FLAX-backed objects
from .utils import dummy_flax_objects
_import_structure["utils.dummy_flax_objects"] = [
   name for name in dir(dummy_flax_objects) if not name.startswith("_")
]

# Direct imports for type-checking
if TYPE_CHECKING:
    # Configuration
    from .configuration_utils import PretrainedConfig

    # Data
    from .data import (
        DataProcessor,
        InputExample,
        InputFeatures,
        SingleSentenceClassificationProcessor,
        SquadExample,
        SquadFeatures,
        SquadV1Processor,
        SquadV2Processor,
        glue_compute_metrics,
        glue_convert_examples_to_features,
        glue_output_modes,
        glue_processors,
        glue_tasks_num_labels,
        squad_convert_examples_to_features,
        xnli_compute_metrics,
        xnli_output_modes,
        xnli_processors,
        xnli_tasks_num_labels,
    )

    # Feature Extractor
    #from .feature_extraction_utils import BatchFeature, PreTrainedFeatureExtractor

    # Files and general utilities
    from .file_utils import (
        CONFIG_NAME,
        MODEL_CARD_NAME,
        PYTORCH_PRETRAINED_BERT_CACHE,
        PYTORCH_TRANSFORMERS_CACHE,
        SPIECE_UNDERLINE,
        TF2_WEIGHTS_NAME,
        TF_WEIGHTS_NAME,
        TRANSFORMERS_CACHE,
        WEIGHTS_NAME,
        TensorType,
        add_end_docstrings,
        add_start_docstrings,
        cached_path,
        is_apex_available,
        is_datasets_available,
        is_faiss_available,
        is_flax_available,
        is_psutil_available,
        is_py3nvml_available,
        is_sentencepiece_available,
        is_sklearn_available,
        is_tf_available,
        is_tokenizers_available,
        is_torch_available,
        is_torch_tpu_available,
    )
    from .hf_argparser import HfArgumentParser

    # Integrations
    #from .integrations import (
    #    is_comet_available,
    #    is_optuna_available,
    #    is_ray_available,
    #    is_ray_tune_available,
    #    is_tensorboard_available,
    #    is_wandb_available,
    #)
    #
    
    from .models.auto import (
        ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CONFIG_MAPPING,
        MODEL_NAMES_MAPPING,
        TOKENIZER_MAPPING,
        AutoConfig,
        AutoTokenizer,
    )
    # ToDo: BERT
    # ToDo:Volta
    
    # Tokenization
    from .tokenization_utils import PreTrainedTokenizer
    from .tokenization_utils_base import (
        AddedToken,
        BatchEncoding,
        CharSpan,
        PreTrainedTokenizerBase,
        SpecialTokensMixin,
        TokenSpan,
    )

    # Trainer
    from .trainer_callback import (
        DefaultFlowCallback,
        EarlyStoppingCallback,
        PrinterCallback,
        ProgressCallback,
        TrainerCallback,
        TrainerControl,
        TrainerState,
    )
    from .trainer_utils import EvalPrediction, IntervalStrategy, SchedulerType, set_seed
    from .training_args import TrainingArguments
    #from .training_args_seq2seq import Seq2SeqTrainingArguments
    #from .training_args_tf import TFTrainingArguments
    
    # Sentencepiece
    from .utils.dummy_sentencepiece_objects import *
    
    # Tokenizers
    from .utils.dummy_tokenizers_objects import *

    # Modeling
    if is_torch_available():

        # Benchmarks
        #from .benchmark.benchmark import PyTorchBenchmark
        #from .benchmark.benchmark_args import PyTorchBenchmarkArguments
        
        from .data.data_collator import (
            DataCollator,
            DataCollatorForLanguageModeling,
            DataCollatorForPermutationLanguageModeling,
            DataCollatorForSeq2Seq,
            DataCollatorForSOP,
            DataCollatorForTokenClassification,
            DataCollatorForWholeWordMask,
            DataCollatorWithPadding,
            default_data_collator,
        )
        from .data.datasets import (
            GlueDataset,
            GlueDataTrainingArguments,
            LineByLineTextDataset,
            LineByLineWithRefDataset,
            LineByLineWithSOPTextDataset,
            SquadDataset,
            SquadDataTrainingArguments,
            TextDataset,
            TextDatasetForNextSentencePrediction,
        )
        from .generation_beam_search import BeamScorer, BeamSearchScorer
        from .generation_logits_process import (
            HammingDiversityLogitsProcessor,
            LogitsProcessor,
            LogitsProcessorList,
            LogitsWarper,
            MinLengthLogitsProcessor,
            NoBadWordsLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
            PrefixConstrainedLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )
        from .generation_utils import top_k_top_p_filtering
        from .modeling_utils import Conv1D, PreTrainedModel, apply_chunking_to_forward, prune_layer
        
        from .models.auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_MAPPING,
            MODEL_WITH_LM_HEAD_MAPPING,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoModelForMultipleChoice,
            AutoModelForNextSentencePrediction,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForTableQuestionAnswering,
            AutoModelForTokenClassification,
            AutoModelWithLMHead,
        )
        # ToDo: BERT
        # ToDo:Volta
        
        # Optimization
        from .optimization import (
            Adafactor,
            AdamW,
            get_constant_schedule,
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup,
            get_linear_schedule_with_warmup,
            get_polynomial_decay_schedule_with_warmup,
            get_scheduler,
        )

        # Trainer
        from .trainer import Trainer
        from .trainer_pt_utils import torch_distributed_zero_first
        #from .trainer_seq2seq import Seq2SeqTrainer
    else:
        from .utils.dummy_pt_objects import *

    # TensorFlow
    # Import the same objects as dummies to get them in the namespace.
    # They will raise an import error if the user tries to instantiate / use them.
    from .utils.dummy_tf_objects import *
    # Import the same objects as dummies to get them in the namespace.
    # They will raise an import error if the user tries to instantiate / use them.
    from .utils.dummy_flax_objects import *

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

        def __getattr__(self, name: str):
            # Special handling for the version, which is a constant from this module and not imported in a submodule.
            if name == "__version__":
                return __version__
            return super().__getattr__(name)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)


if not is_torch_available():
    logger.warning(
        "PyTorch have not been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
