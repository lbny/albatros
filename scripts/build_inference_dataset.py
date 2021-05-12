"""
Author: Lucas Bony

Prepare datasets for inference :
-train
-test
"""
import os.path as osp

import argparse

import numpy as np
import pandas as pd

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument('--train_n_rows', type=int, default=10)
parser.add_argument('--test_n_rows', type=int, default=10)
parser.add_argument('--target_folder', type=str, default='')
parser.add_argument('--train_filepath', type=str, default='train.csv')
parser.add_argument('--test_filepath', type=str, default='test.csv')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

train_df: pd.DataFrame = pd.read_csv(args.train_filepath)
train_df = train_df[['target', 'excerpt']].rename(columns={'target': 'label'})[:args.train_n_rows]
#train_df['label'] = train_df['label'] > -1
test_df: pd.DataFrame = pd.read_csv(args.test_filepath)
test_df = test_df[['excerpt']][:args.test_n_rows]

train_df.to_csv(osp.join(args.target_folder, 'train.csv'), index=False)
test_df.to_csv(osp.join(args.target_folder, 'test.csv'), index=False)



