#!/bin/bash

source_dir=./source/preprocessing/lu_classifier
data_dir=./data/preprocessing/lu_classifier

n_splits=5


modes=(distinct random)

for mode in "${modes[@]}";do
    uv run python ${source_dir}/make_datasets.py \
        --n_splits ${n_splits} \
        --output_dir "${data_dir}/dataset/${mode}" \
        --mode "${mode}"
done