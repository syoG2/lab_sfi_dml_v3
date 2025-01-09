#!/bin/bash

source_dir=./source/verb_clustering_c4
data_dir=./data/verb_clustering_c4/dataset
input_dir=./data/preprocessing/framenet/preprocess

setting_prefix=all
n_splits=3

uv run python ${source_dir}/make_dataset.py \
    --input_file ${input_dir}/exemplars.jsonl \
    --output_dir ${data_dir} \
    --setting_prefix ${setting_prefix} \
    --n_splits ${n_splits} \
    --c4_rate 1
