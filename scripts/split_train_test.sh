#!/usr/bin/bash
python split_train_test.py \
    --train_filepath $HOME/albatros/datasets/train.csv \
    --train_ratio 0.01 \
    --valid_ratio 0.01 \
    --target_folder data/