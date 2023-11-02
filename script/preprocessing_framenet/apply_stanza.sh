#!/bin/bash

source_dir=../../source/preprocessing_framenet
data_dir=../../data/preprocessing_framenet

CUDA=0

CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${source_dir}/apply_stanza.py \
    --input_file ${data_dir}/exemplars/exemplars.jsonl \
    --output_dir ${data_dir}/stanza
