# Mandatory : train_file/validation_file or a task name,
# output_dir
python run_regression.py \
    --model_name_or_path  bert-base-cased \
    --train_file ./data/train.csv \
    --validation_file ./data/valid.csv \
    --test_file ./data/inference/test.csv \
    --inference_file ./data/inference/inference.csv \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --output_dir ./output/ \
    --print_loss_every_steps 3 \
    --n_folds 3