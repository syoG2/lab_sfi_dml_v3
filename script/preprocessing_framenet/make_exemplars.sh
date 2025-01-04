#!/bin/bash

source_dir=./source/preprocessing_framenet
data_dir=./data/preprocessing_framenet

uv run python ${source_dir}/make_exemplars.py \
    --output_dir ${data_dir}/exemplars
