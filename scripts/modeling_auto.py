"""
Author: Lucas Bony
"""
from modeling_boosting import train_one_lightgbm
from typing import Dict

from sklearn.preprocessing import KBinsDiscretizer

import datasets

from modeling_bert import train_one_bert

def quantize(scalar_series, n_bins: int, strategy: str='uniform'):
    """Discretize the continuous variable"""
    
    
    pass
def train_one_model(model_name: str, raw_datasets: datasets.Dataset, args: Dict, logger, 
test_dataset: datasets.Dataset=None, inference_dataset: datasets.Dataset=None, accelerator=None, wandb_tag: str=''):
    """
    Front API to train a model
    """
    if args.n_regression_bins > 0:
        assert args.n_regression_bins > 1, "Cannot have only 1 bin"
        target_binarizer: KBinsDiscretizer = KBinsDiscretizer(n_bins=args.n_regression_bins, strategy=args.bins_strategy, encode='ordinal')
        raw_datasets['train']['label'] = target_binarizer.fit_transform(np.asarray(raw_datasets['train']['label']).reshape(-1, 1))


    if 'bert' in model_name:
        return train_one_bert(raw_datasets, args, logger, test_dataset, inference_dataset, accelerator=accelerator, wandb_tag=wandb_tag, target_binarizer=target_binarizer)

    elif 'gb' in model_name or 'linear' in model_name:
        return train_one_lightgbm(raw_datasets, args, logger, test_dataset, inference_dataset, accelerator=accelerator, wandb_tag=wandb_tag, target_binarizer=target_binarizer)
    
    else:
        raise Exception(f"Unknown model: {model_name}")