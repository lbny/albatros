#!/usr/bin/bash
python build_inference_dataset.py \
    --train_filepath ../datasets/train.csv \
    --test_filepath ../datasets/test.csv \
    --target_folder ./data/inference/ \
    --train_n_rows 10