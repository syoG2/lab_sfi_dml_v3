#!/bin/bash

source_dir=./source/preprocessing
data_dir=./data/preprocessing/framenet/collect


uv run python ${source_dir}/collect_framenet.py\
    --output_file ${data_dir}/exemplars.jsonl