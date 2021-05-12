# Mandatory : train_file/validation_file or a task name,
# output_dir
python run_classification.py \
    --model_name_or_path  bert-base-cased \
    --train_file ./data/train.csv \
    --validation_file ./data/valid.csv \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --output_dir ./output/ \
    --print_loss_every_steps 3