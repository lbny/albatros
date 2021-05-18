"""
Author: Lucas Bony
"""

import os.path as osp

import argparse

import numpy as np
import pandas as pd



import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument('--source_filepath', type=str, default='train.csv')
parser.add_argument('--destination_filepath', type=str, default='test.csv')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

