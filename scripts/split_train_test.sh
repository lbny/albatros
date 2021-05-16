#!/usr/bin/bash
python split_train_test.py \
    --train_filepath $HOME/albatros/datasets/train.csv \
    --train_ratio 0.8 \
    --valid_ratio 0.2 \
    --target_folder data/