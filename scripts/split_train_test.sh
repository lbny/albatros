#!/usr/bin/bash
python split_train_test.py \
    --train_filepath $HOME/projects/albatros/datasets/train.csv \
    --train_ratio 0.1 \
    --valid_ratio 0.1 \
    --target_folder data/