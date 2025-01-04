#!/bin/bash

source_dir=./source/preprocessing/lu_classifier
data_dir=./data/preprocessing/lu_classifier

pretrained_model=bert-base-uncased
n_splits=5

# text_input_style=sep
# text_input_style=token0
text_input_style=token00

for ((part=0; part<n_splits; part++));
do
uv run python ${source_dir}/lu_classifier.py \
    --part "${part}" \
    --pretrained_model "${pretrained_model}" \
    --output_model_dir "${data_dir}/models/${pretrained_model}/${text_input_style}/distinct/${n_splits}_${part}" \
    --device "cuda:2" \
    --mode "crossvalidation_distinct" \
    --text_input_style "${text_input_style}"
done