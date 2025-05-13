#!/bin/bash
# set -ex
set -e

which torchrun

cd LLaMA-Factory

pip install --no-deps -e .
pip install peft==0.11.1 transformers==4.45 accelerate==0.34.0 trl==0.8.6
pip install torchvision


export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
export MASTER_PORT=10523
OUTPUT_DIR=checkpoint/qwen2/sft/hotpot-wikihop

export FORCE_TORCHRUN=1

mkdir -p  ${OUTPUT_DIR}/logs

# template is not important becuase we have already tokenized the dataset

llamafactory-cli train \
    --model_name_or_path hf_models/Qwen2.5-7B-Instruct \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --lora_target all \
    --dataset identity \
    --tokenized_path construct/sft/tokenized_dataset \
    --template qwen25 \
    --cutoff_len 4096 \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 1 \
    --save_steps 500 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --val_size 0 \
    --per_device_eval_batch_size 1 \
    --eval_strategy no \
    --eval_steps 500 \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --report_to tensorboard \
    | tee ${OUTPUT_DIR}/logs/${NODE_RANK}.log

