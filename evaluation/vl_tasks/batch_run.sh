#!/bin/bash
# Batch run script for the GLUE Benchmark
# To set visible GPUs, add CUDA_VISIBLE_DEVICES=X when calling this command:
# CUDA_VISIBLE_DEVICES=0,1 evalutation/batch_run.sh
# This script iterates training num_prefixes x num_tasks x num_models times

# Settings
# - Tasks to be used
tasks="nlvr2"

# - Models to be trained
pt_models="ctrl_lxmert ctrl_uniter ctrl_vilbert ctrl_visual_bert ctrl_vl_bert"
reinit_models="ctrl_lxmert_reinit ctrl_uniter_reinit ctrl_vilbert_reinit ctrl_visual_bert_reinit ctrl_vl_bert_reinit"
models="$pt_models $reinit_models"

# - Prefix for the run (should be integers)
prefixes="0 1 2"

# Debug settings
#tasks="nlvr2"
#models="ctrl_lxmert"
#prefixes="0"

# Directories
pretrained_models_dir="vl_models/pretrained"
output_dir_base="vl_models/vl_finetuned"
task_dir_base="download"
mkdir -p $output_dir_base

for prefix in $prefixes ; do
    for task_name in $tasks ; do
        for model in $models ; do
            
            seed=$(( prefix + 42 ))
            output_dir="$output_dir_base/$prefix/$model/$task_name"
            model_path="$pretrained_models_dir/$model"
            
            echo "$prefix $task_name $model > $output_dir"
            python -u evaluation/vl_tasks/run_vl.py \
                --model_name_or_path "$model_path" \
                --task_name "$task_name" \
                --task_dir "$task_dir_base/$task_name" \
                --do_train \
                --do_eval \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --max_seq_length 128 \
                --per_device_train_batch_size 64 \
                --per_device_eval_batch_size 128 \
                --learning_rate 2e-5 \
                --num_train_epochs 5 \
                --output_dir "$output_dir" \
                --fp16 \
                --disable_tqdm 1 \
                --seed $seed
        done
    done
done
