# eval\_vl\_glue/evaluation

Directory for our evalutaion experiments.

- **glue_tasks** fine-tunes the pretrained models on the GLUE tasks.
- **analysis** summarizes the fine-tuning results
- **vl_tasks** fine-tunes the pretrained models on the NLVR2, used for the sanity check.

## glue_tasks

- **run_glue.py** is a script to fine-tune a model on the GLUE task.
    We modified [run_glue.py](https://github.com/huggingface/transformers/blob/v4.4.0/examples/text-classification/run_glue.py) in the Huggingface's transformers repository to support prediction with the validation sets and multiple validation sets (for MNLI matched and mismatched sets).

    The usage is basically the same as the original one.
    
    ```
    # Run from the repository root
    CUDA_VISIBLE_DEVICES=0 python -u evaluation/glue_tasks/run_glue.py \
        --model_name_or_path vl_models/pretrained/ctrl_vlibert \
        --task_name wnli \
        --do_train \
        --do_eval \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --max_seq_length 128 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 128 \
        --learning_rate 2e-5 \
        --num_train_epochs 5 \
        --output_dir vl_models/finetuned/0/ctrl_vlibert/wnli \
        --fp16 \
        --disable_tqdm 1 \
        --seed 42
    ```

    This is an example of a run where ctrl_vlibert will be pre-trained on wnli and five checkpoints will be made in vl_models/finetuned/0/ctrl_vlibert/wnli.

    For pretrained models, we can get prediction files on the validation sets with the --do_dump_val option.

    ```
    # Run from the repository root and make sure that vl_models/prediction exists.
    python -u python -u evaluation/glue_tasks/run_glue.py \
        --model_name_or_path vl_models/finetuned/0/ctrl_vlibert/wnli \
        --task_name wnli \
        --do_dump_val 1 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --max_seq_length 128 \
        --per_device_train_batch_size 1024 \
        --per_device_eval_batch_size 1024 \
        --dataloader_num_workers 1 \
        --learning_rate 2e-5 \
        --num_train_epochs 5 \
        --output_dir vl_models/prediction/0/ctrl_vlibert/wnli \
        --fp16 \
        --disable_tqdm 1 \
        --seed 42
    ```

- **batch_run.sh** is a batch version of run_glue.py to fine-tune with each model and task repeatedly.
    Run from the repository root directory.

    ```
    CUDA_VISIBLE_DEVICES=0 evaluation/glue_tasks/batch_run.sh
    ```

## analysis

We used Jupyter Notebook.
Please see each notebook for the detail.

- **word_overlap.ipynb** calculates word overlap between corpora (Figure 1)
- **get_glue_score.ipynb** calculates and summarizes models' GLUE score (Table 2 and 5)
- **weight_similarity.ipynb** calculates weight similarity between models (Table 3)
- **error_analysis.ipynb** devides problems according to whether the models are successful in the problems or not (Table 4)

When you run those notebooks in your environment, install the Notebook packages:

```
pip install notebook ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## vl_tasks

As a quick check for our conversion, we trained the models on NLVR2.  
We used image features in the pre-processed data distributed in [e-bug/volta/data](https://github.com/e-bug/volta/tree/main/data).

### Data creation

Before the nlvr2 training, the data creation is required:

1. Make the nlvr2 directory in the vl_tasks directory.
2. Download annotation data (dev.json, test1.json, and train.json) from [lil-lab/nlvr/nlvr2/data/](https://github.com/lil-lab/nlvr/tree/master/nlvr2/data) into vl_tasks/nlvr2/annotations
3. Download image the pre-processed data from [e-bug/volta/data](https://github.com/e-bug/volta/tree/main/data) as vl_tasks/nlvr2/nlvr2-lmdb-imgfeat
4. Convert the pre-processed data into pickle files.  
(Because using the lmdb was slow in our environment, we decided to convert it.)

    ```
    python convert_lmdb.py --src nlvr2/nlvr2-lmdb-imgfeat --dest pickled
    ```


After that data creation, the directory structure will be like below:

```
vl_tasks
    + nlvr2
        + annotations
            - dev.json
            - test1.json
            - train.json
        + pickled
            + train
                (omitted)
            + test1
                (omitted)
            + dev
                (omitted)
        + nlvr2-lmdb-imgfeat
            (omitted)
    - batch_run.sh
    - convert_lmdb.py
    - dataset_nlvr2.py
    - run_vl.py
```

The dataset_nlvr2.py script loads data assuming this structure.

### Training

**run_vl.py**. We modified [run_glue.py](https://github.com/huggingface/transformers/blob/v4.4.0/examples/text-classification/run_glue.py) in the Huggingface's transformers repository to support image feature replacement in the mini-batch creation (using set_format of the Dataset class).  
This shows an example of the usage for ctrl_vlibert

```
python -u evaluation/vl_tasks/run_vl.py \
    --model_name_or_path vl_models/pretrained/ctrl_vlibert \
    --task_name nlvr2 \
    --task_dir evaluation/vl_tasks/nlvr2 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --max_seq_length 128 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --output_dir vl_models/vl_finetuned/0/ctrl_vlibert/nlvr2 \
    --fp16 \
    --disable_tqdm 1 \
    --seed 42
```

**batch_run.sh** is a batch version of run_vl.py.

### Results

We obtained the simmilar training results to those in the Table 2 in [the volta paper](https://arxiv.org/abs/2011.15124).  
We fine-tuned the models by 20 epochs and show the result of the epoch where the models achieved the best *eval_accuracy*.

| model | task | epoch | eval_loss | \*eval_accuracy |
| ----- | ---- | ----- | --------- | --------------- |
| ctrl_lxmert | nlvr2 | 16.0 | 1.708 | 0.694 |
| ctrl_uniter | nlvr2 | 10.0 | 0.937 | 0.722 |
| ctrl_vilbert | nlvr2 | 18.0 | 2.042 | 0.724 |
| ctrl_visual_bert | nlvr2 | 19.0 | 1.702 | 0.720 |
| ctrl_vl_bert | nlvr2 | 10.0 | 0.899 | 0.724 |

