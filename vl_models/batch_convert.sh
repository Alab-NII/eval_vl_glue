#!/bin/bash

python vl_models/convert.py -c download/config/ctrl_visualbert_base.json -w download/volta_weights/ctrl_visual_bert

python vl_models/convert.py -c download/config/ctrl_uniter_base.json -w download/volta_weights/ctrl_uniter

python vl_models/convert.py -c download/config/ctrl_vl-bert_base.json -w download/volta_weights/ctrl_vl_bert

python vl_models/convert.py -c download/config/ctrl_lxmert.json -w download/volta_weights/ctrl_lexmert

python vl_models/convert.py -c download/config/ctrl_vilbert_base.json -w download/volta_weights/ctrl_vilbert
