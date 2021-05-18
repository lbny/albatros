"""
Author: Lucas Bony

Train light gbm regressor.
"""
import gc
from typing import Dict, List, Callable, Union

import numpy as np
import pandas as pd

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
        try:
            vocab = GloVe('6B', vectors_cache='../.vector_cache', dim=args.embeddings_dim)
        except:
            vocab = GloVe('6B', dim=args.embeddings_dim)

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

    def get_sentence_embedding_function(args, vocab, tokenizer, reduction: str='mean') -> Callable:
        
        def glove_embed(sentence: str) -> torch.Tensor:
            """Embed a sentence using provided vocabulary"""
            if reduction == 'mean':
                return torch.mean(vocab.get_vecs_by_tokens(tokenizer(sentence)), axis=0).detach().cpu().numpy()
            elif reduction == 'sum':
                return torch.sum(vocab.get_vecs_by_tokens(tokenizer(sentence)), axis=0).detach().cpu().numpy()
        
        if args.embeddings == 'glove':
            embed_sentence = glove_embed

        return embed_sentence

    pandarallel.initialize(nb_workers=8, progress_bar=True)
    #print(raw_datasets['train'][sentence1_key])
    train_samples_embeddings: List[torch.Tensor] = np.asarray(pd.Series(raw_datasets['train'][sentence1_key]).parallel_apply(
        get_sentence_embedding_function(args, vocab, tokenizer)).tolist())
    valid_samples_embeddings: List[torch.Tensor] = np.asarray(pd.Series(raw_datasets['validation'][sentence1_key]).parallel_apply(
        get_sentence_embedding_function(args, vocab, tokenizer)).tolist())

    assert valid_samples_embeddings.shape[1] == train_samples_embeddings.shape[1], f"Validation and train embedding dim mismatch: {valid_samples_embeddings.shape[1]} and {train_samples_embeddings.shape[1]}"

    if args.test_file:
        test_samples_embeddings: List[torch.Tensor] = np.asarray(pd.Series(test_dataset['test'][sentence1_key]).parallel_apply(
        get_sentence_embedding_function(args, vocab, tokenizer)).tolist())
        assert test_samples_embeddings.shape[1] == train_samples_embeddings.shape[1], f"Test and train embedding dim mismatch: {test_samples_embeddings.shape[1]} and {train_samples_embeddings.shape[1]}"


    if args.inference_file:
        inference_samples_embeddings: List[torch.Tensor] = np.asarray(pd.Series(inference_dataset['inference'][sentence1_key]).parallel_apply(
        get_sentence_embedding_function(args, vocab, tokenizer)).tolist())
        assert inference_samples_embeddings.shape[1] == train_samples_embeddings.shape[1], f"Inference and train embedding dim mismatch: {inference_samples_embeddings.shape[1]} and {train_samples_embeddings.shape[1]}"

    # Training
    # --------
    if args.model_name_or_path == 'lgbm':
        print('DEBUG1')
        train_data: lgbm.Dataset = lgbm.Dataset(np.asarray(train_samples_embeddings), label=train_labels)
        valid_data: lgbm.Dataset = lgbm.Dataset(np.asarray(valid_samples_embeddings), label=valid_labels)

        parameters: Dict = {
            'objective': 'regression',
            'learning_rate': args.learning_rate,
            'num_leaves': args.num_leaves,
            'max_depth': args.max_depth,
            'min_data_in_leaf': args.min_data_in_leaf,
            'feature_fraction': args.feature_fraction,
            'bagging_fraction': args.bagging_fraction,
            'num_iterations': args.num_iterations,
            'metric': 'rmse'
        }

        model = lgbm.train(
            parameters,
            train_data,
            valid_sets=valid_data,
            num_boost_round=args.num_train_epochs
        )

    # Infering
    # --------

    if args.test_file:
        test_predictions = model.predict(test_samples_embeddings)
        output['test_predictions'] = test_predictions

    if args.inference_file:
        inference_predictions = model.predict(inference_samples_embeddings)
        output['inference_predictions'] = inference_predictions

    return output

