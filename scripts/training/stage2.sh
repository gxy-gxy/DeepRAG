#!/bin/bash
# set -ex
set -e

cd LLaMA-Factory

pip install --no-deps -e .
pip install peft==0.11.1 transformers==4.45 accelerate==0.34.0 trl==0.8.6
pip install torchvision


export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
export MASTER_PORT=10523
OUTPUT_DIR=checkpoint/qwen2/dpo/hotpot-wikihop

export FORCE_TORCHRUN=1

mkdir -p  ${OUTPUT_DIR}/logs

# template is not important becuase we have already tokenized the dataset

llamafactory-cli train \
    --model_name_or_path checkpoint/qwen2/sft/hotpot-wikihop \
    --stage dpo \
    --pref_beta 0.4 \
    --do_train true \
    --finetuning_type full \
    --dataset identity \
    --tokenized_path construct/dpo/tokenized_dataset \
    --template llama3 \
    --cutoff_len 4096 \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --output_dir $OUTPUT_DIR \
    --logging_steps 1 \
    --save_strategy epoch \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5.0e-7 \
    --num_train_epochs 8.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --val_size 0.1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy no \
    --eval_steps 500 \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --report_to tensorboard \
    | tee ${OUTPUT_DIR}/logs/${NODE_RANK}.log
