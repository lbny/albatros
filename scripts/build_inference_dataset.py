"""
Author: Lucas Bony

Prepare datasets for inference :
-train
-test

Create two datasets :
test.csv containing only test texts
inference.csv containing both test and train texts
"""
import os.path as osp

import argparse

import numpy as np
import pandas as pd

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument('--train_n_rows', type=int, default=-1)
parser.add_argument('--test_n_rows', type=int, default=-1)
parser.add_argument('--target_folder', type=str, default='')
parser.add_argument('--train_filepath', type=str, default='train.csv')
parser.add_argument('--test_filepath', type=str, default='test.csv')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

# Training dataset
# ----------------
train_df: pd.DataFrame = pd.read_csv(args.train_filepath)
train_df = train_df[['target', 'excerpt']].rename(columns={'target': 'label'})

if args.train_n_rows > 0:
    train_df = train_df[:args.train_n_rows]


# Testing dataset
# ---------------
test_df: pd.DataFrame = pd.read_csv(args.test_filepath)
test_df = test_df[['excerpt']]

if args.test_n_rows > 0:
    test_df = test_df[:args.test_n_rows]

inference_df = pd.concat([train_df[['excerpt']], test_df[['excerpt']]])
inference_df.to_csv(osp.join(args.target_folder, 'inference.csv'), index=False)

test_df.to_csv(osp.join(args.target_folder, 'test.csv'), index=False)



