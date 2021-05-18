# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
from typing import Dict, List, Union, Callable, Tuple

import argparse
import logging
import math
import os
import os.path as osp
import gc
import random

import numpy as np
import torch

import wandb

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from modeling_auto import train_one_model
from utils import format_filepath

logger = logging.getLogger(__name__)

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
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--inference_file", type=str, default=None, help="A csv or a json file containing the inference data."
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Weights and Biases project."
    )
    parser.add_argument(
        "--wandb_name", type=str, default=None, help="Weights and Biases notes."
    )
    parser.add_argument(
        "--wandb_notes", type=str, default=None, help="Weights and Biases notes."
    )
    parser.add_argument(
        "--wandb_tags", type=str, default=None, help="Weights and Biases tags."
    )
    parser.add_argument(
        "--n_folds", type=int, default=1, help="Number of folds to train models on."
    )
    parser.add_argument(
        "--run_name", type=str, default='', help="Name appedned to models."
    )
    parser.add_argument(
        "--tokenizer", type=str, default='basic_english', help="Tokenizer (only in boosting, custom NN or linear mode)."
    )
    parser.add_argument(
        "--embeddings", type=str, default='glove', help="Pretrained embeddings to use (only in boosting, custom NN or linear mode)."
    )
    parser.add_argument(
        "--embeddings_dim", type=int, default=300, help="Dimension of the pretrained embeddings (only in boosting, custom NN or linear mode)."
    )
    parser.add_argument(
        "--bagging_fraction", type=float, default=1, help="Sampling rate of rows in boosting."
    )
    parser.add_argument(
        "--feature_fraction", type=float, default=1, help="Sampling rate of columns in boosting."
    )
    parser.add_argument(
        "--min_data_in_leaf", type=int, default=31, help="Minimum rows to create new leaf."
    )
    parser.add_argument(
        "--max_depth", type=int, default=-1, help="Maximum depth of trees."
    )
    parser.add_argument(
        "--num_iterations", type=int, default=100, help="Number of iterations in boosting."
    )
    parser.add_argument(
        "--alpha", type=float, default=0, help="L1 coef."
    )
    parser.add_argument(
        "--lambda", type=float, default=1, help="L2 coef."
    )
    parser.add_argument(
        "--alpha_1", type=float, default=1e-6, help="Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter."
    )
    parser.add_argument(
        "--lambda_1", type=float, default=1e-6, help="Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter."
    )
    parser.add_argument(
        "--alpha_2", type=float, default=1e-6, help="Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter."
    )
    parser.add_argument(
        "--lambda_2", type=float, default=1e-6, help="Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter."
    )
    parser.add_argument(
        "--sublinear_tf", action='store_true', help="Use sublinear term frequency."
    )
    parser.add_argument(
        "--smooth_idf", action='store_true', help="Smooth idf."
    )
    parser.add_argument(
        "--min_ngram", type=int, default=1, help="Min ngram to use."
    )
    parser.add_argument(
        "--max_ngram", type=int, default=1, help="Max ngram to use."
    )
    parser.add_argument(
        "--max_df", type=float, default=1, help="Max document frequency to use."
    )
    parser.add_argument(
        "--min_df", type=float, default=1, help="Min document frequency to use."
    )
   
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--per_device_inference_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the inference dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--print_loss_every_steps",
        type=int,
        default=None,
        help="Print the loss in training every n steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--eval_metrics_file", type=str, default=None, help="Name of the file containing the metrics, one per epoch."
    )
    parser.add_argument(
        "--save_model", action='store_true', help="Do save model."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if args.inference_file is not None:
            extension = args.inference_file.split(".")[-1]
            assert extension in ["csv", "json"], "`inference_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    if args.wandb_project:

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity='lbny1928',
            notes=args.wandb_notes,
            tags=args.wandb_tags,
            config=args
        )

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

        if args.test_file is not None:
            test_dataset = load_dataset(extension, data_files={'test': args.test_file})
        else:
            test_dataset = None

        if args.inference_file is not None:
            inference_dataset = load_dataset(extension, data_files={'inference': args.inference_file})
        else:
            inference_dataset = None

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if args.n_folds > 1:

        fold_output: List = list()

        kf = KFold(n_splits=args.n_folds)

        indices: np.ndarray = np.arange(len(raw_datasets['train']))

        for fold_id, (train_index, valid_index) in enumerate(kf.split(indices)):
            print(f'Fold {fold_id}')
            # Build fold dataset
            fold_raw_datasets = load_dataset(extension, data_files={'full': data_files['train']})
            fold_raw_datasets['train'] = datasets.Dataset.from_dict(fold_raw_datasets['full'][train_index])
            fold_raw_datasets['validation'] = datasets.Dataset.from_dict(fold_raw_datasets['full'][valid_index])
            output: Dict = train_one_model(args.model_name_or_path, fold_raw_datasets, args, logger, test_dataset, inference_dataset, accelerator=accelerator, wandb_tag=f'fold_{fold_id}')
            # Save fold predictions
            if args.test_file:
                np.save(
                    format_filepath(osp.join(args.output_dir, f'test_predictions_fold_{fold_id}.npy'), args.run_name),
                    output['test_predictions']
                )
            
            if args.inference_file:
                np.save(
                    format_filepath(osp.join(args.output_dir, f'inference_predictions_fold_{fold_id}.npy'), args.run_name),
                    output['inference_predictions']
                )

            fold_output.append(output)

        # Save averaged predictions
        if args.test_file:

            np.save(
                    format_filepath(osp.join(args.output_dir, f'test_predictions.npy'), args.run_name),
                    np.mean([output['test_predictions'] for output in fold_output], axis=0)
                )
        if args.inference_file:

            np.save(
                    format_filepath(osp.join(args.output_dir, f'inference_predictions.npy'), args.run_name),
                    np.mean([output['inference_predictions'] for output in fold_output], axis=0)
                )

    else:
        output: Dict = train_one_model(args.model_name_or_path, raw_datasets, args, logger, test_dataset, inference_dataset, accelerator=accelerator)

    print(output)

if __name__ == "__main__":
    main()