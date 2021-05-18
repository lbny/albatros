"""
Author: Lucas Bony
"""
from typing import List, Dict
import os.path as osp

import argparse

import numpy as np
import pandas as pd

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

from pandarallel import pandarallel

from datasets import load_dataset

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument('--source_filepath', type=str, default='train.csv')
parser.add_argument('--split', type=str, default='')
parser.add_argument('--destination_filepath', type=str, default='test.csv')
parser.add_argument('--p_synonym', type=float, default=0)
parser.add_argument('--p_antonym', type=float, default=0)
parser.add_argument('--p_delete', type=float, default=0)
parser.add_argument('--p_swap', type=float, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--nb_workers', type=int, default=1)

args = parser.parse_args()

pandarallel.initialize(nb_workers=args.nb_workers)

# Loading dataset
# ---------------
dataset = load_dataset('csv', data_files=args.source_filepath)

if args.split:
    dataset = dataset[args.split]

def augmentation_factory(augmentation_name: str):
    """Factory function of augmentation objects from nlpaug"""
    if augmentation_name == 'synonym':
        return naw.SynonymAug(aug_src='wordnet')
    elif augmentation_name == 'antonym':
        return naw.AntonymAug()
    elif augmentation_name == 'swap':
        return naw.RandomWordAug(action="swap")
    elif augmentation_name == 'delete':
        return naw.RandomWordAug()

AUGMENTATIONS: List[str] = ["synonym", "antonym", "swap", "delete"]

print(dir(args.__dict__))

for augmentation in AUGMENTATIONS:

    if args.__dict__[f"p_{augmentation}"] > 0.0:
        augmenter = augmentation_factory(augmentation)
        sample_df = pd.DataFrame(dataset).sample(frac=args.__dict__[f"p_{augmentation}"], replace=False)
        sample_df['augmented_text'] = sample_df["excerpt"].parallel_apply(lambda x: augmenter.augment([x]))




