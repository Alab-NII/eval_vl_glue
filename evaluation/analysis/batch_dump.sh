#!/bin/bash

MODE=exec

# We assume that working directory is the analysis directory
SCRIPT_PATH='../glue_tasks/run_glue.py'

TARGET_LIST_FILE="$1"
OUTPUT_ROOT_DIR='prediction'
SEED=42 # not used

cat $TARGET_LIST_FILE | while read line ; do
    
    # skip lines startswith '#' (header)
    if [ "${line:0:1}" = "#" ]; then
         continue
    fi
    
    # make path
    model_name=$(echo $line | awk -F" " '{print $1}')
    trial=$(echo $line | awk -F" " '{print $2}')
    task_name=$(echo $line | awk -F" " '{print $3}')
    checkpoint_path=$(echo $line | awk -F" " '{print $7}')
    output_dir="$OUTPUT_ROOT_DIR/$model_name/$task_name/$trial"
    
    # execute
    echo $checkpoint_path '->' $output_dir
    if [ $MODE = 'exec' ] ; then
        python -u "$SCRIPT_PATH" \
            --model_name_or_path "$checkpoint_path" \
            --task_name "$task_name" \
            --do_dump_val 1 \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --max_seq_length 128 \
            --per_device_train_batch_size 1024 \
            --per_device_eval_batch_size 1024 \
            --dataloader_num_workers 1 \
            --learning_rate 2e-5 \
            --num_train_epochs 5 \
            --output_dir "$output_dir" \
            --fp16 \
            --disable_tqdm 1 \
            --seed $SEED
    fi
done

