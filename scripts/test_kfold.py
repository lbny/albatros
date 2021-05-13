from sklearn.model_selection import KFold

import numpy as np
from datasets import load_dataset
import datasets

kf = KFold(n_splits=5)
dataset = load_dataset('csv', data_files={'train': '/home/lbony/projects/albatros/scripts/data/train.csv'})
indices: np.ndarray = np.arange(len(dataset['train']))

for fold_id, (train_index, valid_index) in enumerate(kf.split(indices)):

    fold_dataset = load_dataset('csv', data_files={'train': '/home/lbony/projects/albatros/scripts/data/train.csv'})
    fold_dataset['train'] = datasets.Dataset.from_dict(fold_dataset['train'][train_index])
    fold_dataset['validation'] = datasets.Dataset.from_dict(fold_dataset['train'][valid_index])

