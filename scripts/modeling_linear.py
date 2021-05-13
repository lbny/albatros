"""
Author: Lucas Bony
"""
from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer

import datasets


def train_one_linear_model(raw_datasets: datasets.Dataset, args: Dict, logger, test_dataset: datasets.Dataset=None, inference_dataset: datasets.Dataset=None, accelerator=None):

    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        n_gram_range=(1,2)
    )