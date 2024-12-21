#!/bin/bash

DATASET_DIR="example_data/"
PRETRAIN_MODEL="/path/to/Skywork-o1-Open-Llama-3.1-8B"
SAVE_DIR="path/to/save_dir"

export CUDA_VISIBLE_DEVICES=0

for FILE in $DATASET_DIR/*.jsonl; do
    FILENAME=$(basename $FILE .jsonl)
    CHECKPOINT_PATH="$SAVE_DIR/${FILENAME}"

    python sft.py \
    --model_name_or_path $PRETRAIN_MODEL \
    --max_seq_length 2048 \
    --dataset_name $FILE \
    --learning_rate 1e-5 \
    --torch_dtype bfloat16 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --weight_decay 1e-4 \
    --eval_strategy no \
    --warmup_steps 10 \
    --output_dir $CHECKPOINT_PATH \
    --save_strategy no \
    --bf16 true \
    --use_peft \
    --lora_r 2 \
    --lora_alpha 16 \
    --lora_dropout 0.1\
    --max_grad_norm 1.0 

done