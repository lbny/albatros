"""
Author: Lucas Bony
"""
from modeling_boosting import train_one_lightgbm
from typing import Dict

from sklearn.preprocessing import KBinsDiscretizer

import numpy as np

import datasets

from modeling_bert import train_one_bert


def train_one_model(model_name: str, raw_datasets: datasets.Dataset, args: Dict, logger, 
test_dataset: datasets.Dataset=None, inference_dataset: datasets.Dataset=None, accelerator=None, wandb_tag: str=''):
    """
    Front API to train a model
    """


    if 'bert' in model_name:
        return train_one_bert(raw_datasets, args, logger, test_dataset, inference_dataset, accelerator=accelerator, wandb_tag=wandb_tag, target_binarizer=target_binarizer)

    elif 'gb' in model_name or 'linear' in model_name:
        return train_one_lightgbm(raw_datasets, args, logger, test_dataset, inference_dataset, accelerator=accelerator, wandb_tag=wandb_tag, target_binarizer=target_binarizer)
    
    else:
        raise Exception(f"Unknown model: {model_name}")