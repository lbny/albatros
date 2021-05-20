"""
Author: Lucas Bony

Train light gbm regressor.
"""
import os.path as osp
import json
import gc
from typing import Dict, List, Callable, Union

import numpy as np
import pandas as pd

from pandarallel import pandarallel
from tqdm import tqdm
tqdm.pandas()

import lightgbm as lgbm
import xgboost as xgb

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csr_matrix

import datasets
import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe, FastText, CharNGram

from wandb.xgboost import wandb_callback

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
            vocab = GloVe('6B', cache=args.embeddings_cache, dim=args.embeddings_dim)
        except:
            vocab = GloVe('6B', dim=args.embeddings_dim)
    elif args.embeddings == 'charngram':
        try:
            vocab = CharNGram(cache=args.embeddings_cache)
        except:
            vocab = CharNGram()
    elif args.embeddings == 'fasttext':
        try:
            vocab = FastText(cache=args.embeddings_cache)
        except:
            vocab = FastText()

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

    if args.embeddings in ['tfidf']:
        tfidf: TfidfVectorizer = TfidfVectorizer(
            sublinear_tf=args.sublinear_tf,
            ngram_range=(args.min_ngram, args.max_ngram),
            smooth_idf=args.smooth_idf,
            max_df=args.max_df,
            min_df=args.min_df
        )
        tfidf.fit(raw_datasets['train'][sentence1_key] + raw_datasets['validation'][sentence1_key] + test_dataset['test'][sentence1_key], inference_dataset['inference'][sentence1_key])
    
        train_samples_embeddings: csr_matrix = tfidf.transform(raw_datasets['train'][sentence1_key])
        valid_samples_embeddings: csr_matrix = tfidf.transform(raw_datasets['validation'][sentence1_key])

        if args.test_file:
            test_samples_embeddings: csr_matrix = tfidf.transform(test_dataset['test'][sentence1_key])

        if args.inference_file:
            inference_samples_embeddings: csr_matrix = tfidf.transform(inference_dataset['inference'][sentence1_key])


    else:
        if args.nb_workers > 1:
            pandarallel.initialize(nb_workers=args.nb_workers, progress_bar=True)

        def apply(sample_df: pd.Series, fun: Callable, nb_workers: int=1) -> pd.Series:
            """Use different functions depending on number of workers"""
            if nb_workers > 1:
                return sample_df.paralllel_apply(fun)
            return sample_df.progress_apply(fun)

        #print(raw_datasets['train'][sentence1_key])
        train_samples_embeddings: List[torch.Tensor] = np.asarray(apply(pd.Series(raw_datasets['train'][sentence1_key]), get_sentence_embedding_function(args, vocab, tokenizer)).tolist())
        valid_samples_embeddings: List[torch.Tensor] = np.asarray(apply(pd.Series(raw_datasets['validation'][sentence1_key]), get_sentence_embedding_function(args, vocab, tokenizer)).tolist())

        assert valid_samples_embeddings.shape[1] == train_samples_embeddings.shape[1], f"Validation and train embedding dim mismatch: {valid_samples_embeddings.shape[1]} and {train_samples_embeddings.shape[1]}"

        if args.test_file:
            test_samples_embeddings: List[torch.Tensor] = np.asarray(apply(pd.Series(test_dataset['test'][sentence1_key]), get_sentence_embedding_function(args, vocab, tokenizer)).tolist())
            assert test_samples_embeddings.shape[1] == train_samples_embeddings.shape[1], f"Test and train embedding dim mismatch: {test_samples_embeddings.shape[1]} and {train_samples_embeddings.shape[1]}"


        if args.inference_file:
            inference_samples_embeddings: List[torch.Tensor] = np.asarray(apply(pd.Series(inference_dataset['inference'][sentence1_key]), get_sentence_embedding_function(args, vocab, tokenizer)).tolist())
            assert inference_samples_embeddings.shape[1] == train_samples_embeddings.shape[1], f"Inference and train embedding dim mismatch: {inference_samples_embeddings.shape[1]} and {train_samples_embeddings.shape[1]}"

    # Training
    # --------
    if args.model_name_or_path == 'lgbm':

        train_data: lgbm.Dataset = lgbm.Dataset(np.asarray(train_samples_embeddings), label=train_labels)
        valid_data: lgbm.Dataset = lgbm.Dataset(np.asarray(valid_samples_embeddings), label=valid_labels)

        if args.num_leaves == -1:
            num_leaves: int = 2 ** args.max_depth
        else:
            num_leaves: int = args.num_leaves

        parameters: Dict = {
            'objective': 'regression',
            'learning_rate': args.learning_rate,
            'num_leaves': num_leaves,
            'max_depth': args.max_depth,
            'min_data_in_leaf': args.min_data_in_leaf,
            'feature_fraction': args.feature_fraction,
            'bagging_fraction': args.bagging_fraction,
            'num_iterations': args.num_train_epochs,
            'metric': 'rmse'
        }

        callbacks: List = []

        if args.wandb_project is not None:
            callbacks.append(wandb_callback())

        model = lgbm.train(
            parameters,
            train_data,
            valid_sets=valid_data,
            callbacks=callbacks
        )

    elif args.model_name_or_path == 'xgboost':
        
        train_data: xgb.DMatrix = xgb.DMatrix(np.asarray(train_samples_embeddings), label=train_labels)
        valid_data: xgb.DMatrix = xgb.DMatrix(np.asarray(valid_samples_embeddings), label=valid_labels)

        tree_method: str = 'hist'
        if torch.cuda.is_available():
            tree_method = 'gpu_hist'

        parameters: Dict = {
            'objective': 'reg:squarederror',
            'learning_rate': args.learning_rate,
            'max_leaves': args.num_leaves,
            'max_depth': args.max_depth,
            'gamma': args.min_data_in_leaf,
            'colsample_bytree': args.feature_fraction,
            'subsample': args.bagging_fraction,
            'num_round': args.num_train_epochs,
            'lambda': args.reg_lambda,
            'alpha': args.reg_alpha,
            'tree_method': tree_method,
            'eval_metric': 'rmse'
        }

        callbacks: List = []

        if args.wandb_project is not None:
            callbacks.append(wandb_callback())

        model = xgb.train(
            params=parameters,
            dtrain=train_data,
            evals=[(train_data, 'train'), (valid_data, 'valid')],
            callbacks=callbacks
        )

        train_samples_embeddings = xgb.DMatrix(np.asarray(train_samples_embeddings))
        valid_samples_embeddings = xgb.DMatrix(np.asarray(valid_samples_embeddings))
        test_samples_embeddings = xgb.DMatrix(np.asarray(test_samples_embeddings))
        inference_samples_embeddings = xgb.DMatrix(np.asarray(inference_samples_embeddings))


    elif args.model_name_or_path == 'linear_bayesian_ridge':
        
        model = BayesianRidge(
            n_iter=args.num_train_epochs,
            alpha_1=args.alpha_1,
            alpha_2=args.alpha_2,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            normalize=args.normalize
        )

        model.fit(train_samples_embeddings, train_labels)        

    elif args.model_name_or_path == 'linear_mse':
        model = LinearRegression()

        model.fit(train_samples_embeddings, train_labels)

        valid_preds = model.predict(valid_samples_embeddings)
        eval_loss = np.sqrt(mean_squared_error(valid_labels, valid_preds))
        print(eval_loss)

    # Validating
    # ----------

    valid_preds = model.predict(valid_samples_embeddings)
    train_preds = model.predict(train_samples_embeddings)
    eval_loss = np.sqrt(mean_squared_error(valid_labels, valid_preds))
    train_loss = np.sqrt(mean_squared_error(train_labels, train_preds))
    print(f"Eval loss: {eval_loss}")
    print(f"Eval loss: {train_loss}")

    # Infering
    # --------

    if args.test_file:
        test_predictions = model.predict(test_samples_embeddings)
        output['test_predictions'] = test_predictions

    if args.inference_file:
        inference_predictions = model.predict(inference_samples_embeddings)
        output['inference_predictions'] = inference_predictions

    if args.eval_metrics_file:        

        with open(osp.join(args.output_dir, '_'.join([args.eval_metrics_file, "0"]) + '.json'), 'w') as output_stream:
            json.dump({'validation_rmse_loss': float(eval_loss),
            'train_rmse_loss': float(train_loss)}, output_stream)
            output_stream.close()

    return output

