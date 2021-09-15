#!/bin/bash

WEIGHTS_DIR=download/volta_weights
CONFIG_DIR=download/volta_config
SCRIPT_PATH=vl_models/convert.py

python $SCRIPT_PATH -c $CONFIG_DIR/ctrl_visualbert_base.json -w $WEIGHTS_DIR/ctrl_visual_bert

python $SCRIPT_PATH -c $CONFIG_DIR/ctrl_uniter_base.json -w  $WEIGHTS_DIR/ctrl_uniter

python $SCRIPT_PATH -c $CONFIG_DIR/ctrl_vl-bert_base.json -w  $WEIGHTS_DIR/ctrl_vl_bert

python $SCRIPT_PATH -c $CONFIG_DIR/ctrl_lxmert.json -w  $WEIGHTS_DIR/ctrl_lxmert

python $SCRIPT_PATH -c $CONFIG_DIR/ctrl_vilbert_base.json -w  $WEIGHTS_DIR/ctrl_vilbert
