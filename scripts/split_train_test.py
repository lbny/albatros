"""
Author: Lucas Bony

Splits train and valid csv
"""
import os.path as osp

import argparse

import numpy as np
import pandas as pd

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--valid_ratio', type=float, default=0.2)
parser.add_argument('--target_folder', type=str, default='')
parser.add_argument('--train_filepath', type=str, default='train.csv')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

train_df: pd.DataFrame = pd.read_csv(args.train_filepath)

train_df = train_df[['target', 'excerpt']].rename(columns={'target': 'label'})
#train_df['label'] = train_df['label'] > -1

train_indices: np.ndarray = np.random.choice(np.arange(train_df.shape[0]), int(args.train_ratio * train_df.shape[0]), replace=False)

avalailable_indices: np.ndarray = np.setdiff1d(np.arange(train_df.shape[0]), train_indices)

valid_indices: np.ndarray = np.random.choice(
    avalailable_indices,
    int(args.valid_ratio * (avalailable_indices.shape[0] / (1 - args.train_ratio))),
    replace=False
)

assert np.intersect1d(train_indices, valid_indices).shape[0] == 0, "Train and valid indices are overlapping"

train_df.iloc[train_indices].to_csv(osp.join(args.target_folder, 'train.csv'), index=False)
train_df.iloc[valid_indices].to_csv(osp.join(args.target_folder, 'valid.csv'), index=False)



