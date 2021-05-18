#!usr/bin/bash
python augment_dataset.py \
    --source_filepath ./data/train.csv \
    --split train \
    --destination_filepath ./data/augmented_train.csv \
    --p_swap 0.1