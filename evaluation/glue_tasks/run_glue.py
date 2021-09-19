#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
# We changed 'transformers' to 'eval_vl_glue.trransformers_volta' for the package imports
# We overwrite Trainer by CustomTrainer (defined around line 168) to pass dict of dataset for evaluation (eval_dataset) 

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric, DownloadConfig

from eval_vl_glue import transformers_volta as transformers
from eval_vl_glue.transformers_volta import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from eval_vl_glue.transformers_volta.trainer_utils import get_last_checkpoint, is_main_process

# For CustomTrainer Class
import time
import collections
import torch
import datasets
from typing import Union, Optional, Dict, Callable, Tuple, List
from eval_vl_glue.transformers_volta.data.data_collator import DataCollator
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from eval_vl_glue.transformers_volta.modeling_utils import PreTrainedModel
from eval_vl_glue.transformers_volta.trainer_utils import EvalPrediction, speed_metrics
from eval_vl_glue.transformers_volta.trainer_callback import TrainerCallback, ProgressCallback


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "ax": ("premise", "hypothesis"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    
    # Add this option to dump prediction for the validation set.
    do_dump_val: Optional[bool] = field(
        default=False, metadata={"help": "Whether to dump prediction for the validation set at last or not"}
    )
    
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

        
class CustomTrainer(Trainer):
    """We overwrite Trainer to pass dict of dataset for evaluation (eval_dataset)"""
    
    @staticmethod
    def _is_dict(x):
        return isinstance(x, (dict, collections.OrderedDict))
    
    def _assert_sized_data(self, data):
        if not self._is_dict(data):
            data = {'':data}
        for k, v in data.items():
            if v is not None and not isinstance(v, collections.abc.Sized):
                _msg = '%s@eval_dataset'%k if k else 'eval_dataset'
                raise ValueError("%s must implement __len__"%_msg)
    
    def _maybe_remove_unused_columns(self, data, description='evaluation'):
        if transformers.file_utils.is_datasets_available():
            if not self._is_dict(data):
                data = {'':data}
            for k, v in data.items():
                if isinstance(v, datasets.Dataset):
                    _desc = '%s@%s'%(k, description) if k else description
                    self._remove_unused_columns(v, description=_desc)
    
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        
        super().__init__(model, args, data_collator, train_dataset, None, 
            tokenizer, model_init, compute_metrics, callbacks, optimizers)
        
        self._assert_sized_data(eval_dataset)
        self._maybe_remove_unused_columns(eval_dataset)
        self.eval_dataset = eval_dataset
    
    def get_eval_dataloader(
        self, 
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
    ) -> Union[DataLoader, Dict[str, DataLoader]]:
        
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        self._assert_sized_data(eval_dataset)
        self._maybe_remove_unused_columns(eval_dataset)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        if self._is_dict(eval_dataset):
            single_loader = False
        else:
            eval_dataset = {'':eval_dataset}
            single_loader = True
        
        loaders = collections.OrderedDict()
        for _dataset_key, _dataset_val in eval_dataset.items():
            eval_sampler = self._get_eval_sampler(_dataset_val)
            loaders[_dataset_key] = DataLoader(
                _dataset_val,
                sampler=eval_sampler,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                 num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
        return loaders[''] if single_loader else loaders
    
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        
        # Search ProgressCallback
        # uses tqdm update
        progress_callback = None
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, ProgressCallback):
                progress_callback = callback
                break
        
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        
        output_metrics = {}
        n_samples = 0
        if not  self._is_dict(eval_dataloader):
            eval_dataloader = {'':eval_dataloader}
        for _loader_key, _loader_val in eval_dataloader.items():
            _desc = _loader_key+'@evaluation' if _loader_key else 'Evaluation'
            _prefix = metric_key_prefix + (('_'+_loader_key) if _loader_key else '')
            output = self.prediction_loop(
                _loader_val,
                description=_desc,
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=_prefix,
            )
            output_metrics.update(output.metrics)
            n_samples += len(_loader_val.dataset)
            
            if progress_callback is not None:
                # We do not end evaluation, but want to switch new tqdm
                # So call just the event of pregress_callback
                progress_callback.on_evaluate(self.args, self.state, self.control)
        
        output_metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.log(output_metrics)
        
        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())
       
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output_metrics)
        
        self._memory_tracker.stop_and_update_metrics(output_metrics)

        return output_metrics


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        download_config = DownloadConfig(local_files_only=False)
        datasets = load_dataset("glue", data_args.task_name, download_config=download_config)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if data_args.task_name == 'ax':
            label_list = ['entailment', 'neutral', 'contradiction'] # same order as mnli
            num_labels = len(label_list)
        
        elif not is_regression:
            if 'train' in datasets:
                label_list = datasets["train"].features["label"].names
                num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
        
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None
    
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    
    _map = lambda x: x.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    #datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = None
    if training_args.do_train:
        train_dataset = _map(datasets["train"])
    
    eval_dataset = None
    if training_args.do_eval or data_args.do_dump_val:
        if data_args.task_name == "mnli":
            eval_dataset = collections.OrderedDict()
            for key, name in [('', "validation_matched"), ('mm', 'validation_mismatched')]:
                eval_dataset[key] = _map(datasets[name])
        else:
            eval_dataset = _map(datasets["validation"])
    
    test_dataset = None
    #if data_args.task_name is not None or data_args.test_file is not None:
    if training_args.do_predict:
        if data_args.task_name == "mnli":
            test_dataset = collections.OrderedDict()
            for key, name in [('', "test_matched"), ('mm', 'test_mismatched')]:
                test_dataset[key] = _map(datasets[name])
        else:
            test_dataset = _map(datasets["test"])
    
    # Log a few random samples from the training set:
    if train_dataset is not None:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None and data_args.task_name != 'ax':
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics
    
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
            # avoid overwriting num_labels from local pretrained model
            #elif os.path.isdir(model_args.model_name_or_path):
            #    checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    def dump_prediction(target_dataset, output_file_prefix, label_attached):
        
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        if data_args.task_name == "mnli":
            tasks = ['mnli_m', 'mnli_mm']
            target_datasets = [target_dataset[''], target_dataset['mm']]
        else:
            tasks = [data_args.task_name]
            target_datasets = [target_dataset]
        
        for target_dataset, task in zip(target_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that (for the test case).
            dataset_wo_label = target_dataset.map(remove_columns=['label'])
            predictions = trainer.predict(test_dataset=dataset_wo_label).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            
            output_test_file = os.path.join(training_args.output_dir, f"{output_file_prefix}_results_{task}.txt")
            if trainer.is_world_process_zero():
                columns = ['index', 'prediction'] + ['label']*label_attached
                with open(output_test_file, "w") as writer:
                    writer.write('\t'.join(columns) + '\n')
                    for index, (pred, data) in enumerate(zip(predictions, target_dataset)):
                        label = data['label']
                        if is_regression:
                            pred = f'{pred:3.3f}'
                            label = f'{label:3.3f}'
                        else:
                            pred = label_list[pred]
                            label = label_list[label] if label >= 0 else None
                        
                        if label_attached:
                            writer.write(f"{index}\t{pred}\t{label}\n")
                        else:
                            writer.write(f"{index}\t{pred}\n")
                            
                logger.info(f"***** {output_file_prefix} results {task} created *****")
    
    if data_args.do_dump_val:
        logger.info("*** Dump validation results ***")
        dump_prediction(eval_dataset, 'valid', label_attached=True)

    if training_args.do_predict:
        logger.info("*** Test ***")
        dump_prediction(test_dataset, 'test', label_attached=False)
    
    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)    
    
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
