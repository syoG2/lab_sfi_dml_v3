#!/bin/bash

source_dir=./source/verb_clustering_c4
data_dir=./data/verb_clustering_c4/dataset
input_dir=./data/preprocessing/framenet/preprocess

setting_prefix=all
n_splits=3

# add_method=c4first
# add_method=ratio
# add_method=sequential

# add_method=c4first_verb
# add_method=ratio_verb
# add_method=sequential_verb
add_method=sequential_n_verb

c4_rate=1

verb_form=original
# verb_form=lemma

uv run python ${source_dir}/make_dataset.py \
    --input_file ${input_dir}/${verb_form}/exemplars.jsonl \
    --output_dir ${data_dir}/${verb_form}/${add_method} \
    --setting_prefix ${setting_prefix} \
    --n_splits ${n_splits} \
    --c4_rate ${c4_rate} \
    --add_method ${add_method}