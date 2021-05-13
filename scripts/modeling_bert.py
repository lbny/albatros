"""
Author: Lucas Bony
"""
from typing import Dict
import os.path as osp
import logging
import math
import os
import gc
import random
import wandb

import numpy as np
import torch

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)


def train_one_bert(raw_datasets: datasets.Dataset, args: Dict, logger, 
test_dataset: datasets.Dataset=None, inference_dataset: datasets.Dataset=None, accelerator=None, wandb_tag: str=''):
        """
        raw_datasets (datasets.Dataset): The main dataset containing two splits : 'train' and 'validation'
        """
        # Labels
        if args.task_name is not None:
            is_regression = args.task_name == "stsb"
            if not is_regression:
                label_list = raw_datasets["train"].features["label"].names
                num_labels = len(label_list)
            else:
                num_labels = 1
        else:
            # Trying to have good defaults here, don't hesitate to tweak to your needs.
            is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
            if is_regression:
                num_labels = 1
            else:
                # A useful fast method:
                # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
                label_list = raw_datasets["train"].unique("label")
                label_list.sort()  # Let's sort it for determinism
                num_labels = len(label_list)

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        # Preprocessing the datasets
        if args.task_name is not None:
            sentence1_key, sentence2_key = task_to_keys[args.task_name]
        else:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
            if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and args.task_name is not None
            and not is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                logger.info(
                    f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                    "Using it!"
                )
                label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif args.task_name is None:
            if is_regression:
                label_to_id = None
            else:
                label_to_id = {v: i for i, v in enumerate(label_list)}

        padding = "max_length" if args.pad_to_max_length else False

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

            if "label" in examples:
                if label_to_id is not None:
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l] for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = examples["label"]
            return result

        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
        )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

        if args.test_file:
            test_dataset = test_dataset.map(
                preprocess_function, batched=True, remove_columns=test_dataset["test"].column_names
            )["test"]

        if args.inference_file:
            inference_dataset = inference_dataset.map(
                preprocess_function, batched=True, remove_columns=inference_dataset["inference"].column_names
            )["inference"]


        # Containing all output data depending on function arguments
        output: Dict = {}

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # DataLoaders creation:
        if args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        if args.test_file:
            test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)

        if args.inference_file:
            inference_dataloader = DataLoader(inference_dataset, collate_fn=data_collator, batch_size=args.per_device_inference_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with our `accelerator`.
        if args.test_file:
            if args.inference_file:
                model, optimizer, train_dataloader, eval_dataloader, test_dataloader, inference_dataloader = accelerator.prepare(
                    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, inference_dataloader
                )
            else:  
                model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
                    model, optimizer, train_dataloader, eval_dataloader, test_dataloader
                )
        else:
            model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader
            )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )


        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(args.num_train_epochs):
            model.train()
            train_loss = []
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps

                if args.print_loss_every_steps:
                    if step % args.print_loss_every_steps == 0:
                        print(f"Train Loss at {step}: {np.sqrt(np.mean(train_loss[-args.print_loss_every_steps:]))}")

                if args.wandb_project:
                    wandb.log({
                        "epoch": epoch,
                        "step": step,
                        "train_rmse_loss": loss.sqrt(),
                        "lr_0": optimizer.param_groups[0]['lr'],
                        "lr_1": optimizer.param_groups[1]['lr'],
                        "tag": wandb_tag
                    })

                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break
                
                train_loss.append(loss.detach().cpu().numpy())
                del loss, outputs, batch
                gc.collect()
            
            train_loss = np.sqrt(np.mean(train_loss))
            print(f'Total Train loss - epoch {epoch} - RMSE {train_loss}')

            if args.wandb_project:
                wandb.log({'epoch': epoch, 'total_train_loss': train_loss, 'tag': wandb_tag})

            model.eval()
            eval_loss = []
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    eval_loss.append(loss.detach().cpu().numpy())
                    if args.print_loss_every_steps:
                        if step % args.print_loss_every_steps == 0:
                            print(f"Validation Loss at {step}: {np.sqrt(np.mean(eval_loss[-args.print_loss_every_steps:]))}")
                    del loss, outputs, batch
                    gc.collect()

                
                eval_loss = np.sqrt(np.mean(eval_loss))    
                print(f'Total Validation RMSE loss : {eval_loss}')

                if args.wandb_project:
                    wandb.log({
                        "epoch": epoch,
                        "validation_rmse_loss": eval_loss,
                        'tag': wandb_tag
                    })

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            del unwrapped_model
            gc.collect()

        if args.task_name == "mnli":
            # Final evaluation on mismatched validation set
            eval_dataset = processed_datasets["validation_mismatched"]
            eval_dataloader = DataLoader(
                eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
            )
            eval_dataloader = accelerator.prepare(eval_dataloader)

            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)
                    del outputs
                    gc.collect()        

        if args.test_file:
            print('Infering on test file...')
            predictions = []
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(test_dataloader):
                    outputs = model(**batch)
                    y_preds = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    # in case batch size is 1
                    if len(y_preds.size()) == 0:
                        y_preds = y_preds.view(1)

                    predictions.append(y_preds.detach())
                    del outputs, y_preds
                    gc.collect()

                #np.save(
                #    osp.join(args.output_dir, 'test_predictions.npy'),
                #    torch.cat(predictions).cpu().numpy()
                #)
                output['test_predictions'] = torch.cat(predictions).cpu().numpy()

        if args.inference_file:
            print('Infering on inference file...')
            predictions = []
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(inference_dataloader):
                    outputs = model(**batch)
                    y_preds = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    # in case batch size is 1
                    if len(y_preds.size()) == 0:
                        y_preds = y_preds.view(1)

                    predictions.append(y_preds.detach())
                    del outputs, y_preds
                    gc.collect()

                output['inference_predictions'] = torch.cat(predictions).cpu().numpy()

                #np.save(
                #    osp.join(args.output_dir, 'inference_predictions.npy'),
                #    torch.cat(predictions).cpu().numpy()
                #)

    

        return output