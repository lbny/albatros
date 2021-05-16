"""
Author: Lucas Bony

Train light gbm regressor.
"""
import gc
from typing import Dict, List, Callable, Union

import numpy as np

from pandarallel import pandarallel

import lightgbm as lgbm

import datasets
import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe

def train_one_lightgbm(raw_datasets: datasets.Dataset, args: Dict, logger, 
    test_dataset: datasets.Dataset=None, inference_dataset: datasets.Dataset=None, accelerator=None, wandb_tag: str=''):
    """
    Train one single lightgbm model for text classification/regression.
    raw_datasets must have two splits : 'train' and 'validation' ; both split must have a 'label' column and at least
    one column containing text.
    """
    output: Dict = {}

    # Tokenizer and vocabulary
    # ------------------------
    tokenizer = get_tokenizer(args.tokenizer)

    if args.embeddings == 'glove':

        vocab = GloVe('6B', dim=args.embeddings_dim)

    vocab.get_vecs_by_tokens(tokenizer('Hello world !'))

    # Text column
    # -----------
    # Determine which column contains the text to process

    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Labels
    # ------
    # Define 'num_labels', the number of labels (1 if regression)
    # and 'is_regression'
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    train_labels: List[Union[float, int, List[int]]] = raw_datasets['train']['label']
    valid_labels: List[Union[float, int, List[int]]] = raw_datasets['validation']['label']

    # Embeddings
    # ----------
    # Compute embeddings for all sentences in train and validation
    # corpuses and store them in lists
    # TODO train using datasets map function ?
    def get_sentence_embedding_function(vocab, tokenizer) -> Callable:
        def embed_sentence(sentence: str) -> torch.Tensor:
            """Embed a sentence using provided vocabulary"""
            return vocab.get_vecs_by_tokens(tokenizer(sentence))

    pandarallel.initialize(nb_workers=8, progress_bar=True)

    train_samples_embeddings: List[torch.Tensor] = raw_datasets['train'][sentence1_key].parallel_apply(
        get_sentence_embedding_function(vocab, tokenizer)).tolist()
    valid_samples_embeddings: List[torch.Tensor] = raw_datasets['validation'][sentence1_key].parallel_apply(
        get_sentence_embedding_function(vocab, tokenizer)).tolist()    

    if args.test_file:
        test_samples_embeddings: List[torch.Tensor] = test_dataset[sentence1_key].parallel_apply(
        get_sentence_embedding_function(vocab, tokenizer)).tolist()

    if args.inference_file:
        inference_samples_embeddings: List[torch.Tensor] = inference_dataset[sentence1_key].parallel_apply(
        get_sentence_embedding_function(vocab, tokenizer)).tolist()


    # Training
    # --------

    if args.model == 'lgbm':

        train_data: lgbm.Dataset = lgbm.Dataset(np.asarray(train_samples_embeddings), label=train_labels)
        valid_data: lgbm.Dataset = lgbm.Dataset(np.asarray(valid_samples_embeddings), label=valid_labels)

        parameters: Dict = {
            'objective': 'regression',
            'learning_rate': 1e-2,
        }

        model = lgbm.train(
            parameters,
            train_data,
            valid_sets=valid_data,
            num_boost_round=args.num_train_epochs,
            early_stopping_rounds=100
        )

    # Infering
    # --------

    if args.test_file:
        test_data: lgbm.Dataset = lgbm.Dataset(np.asarray(test_samples_embeddings))
        test_predictions = model.predict(test_data)
        output['test_predictions'] = test_predictions

    if args.inference_file:
        inference_data: lgbm.Dataset = lgbm.Dataset(np.asarray(inference_samples_embeddings))
        inference_predictions = model.predict(inference_data)
        output['inference_predictions'] = inference_predictions

    return output

