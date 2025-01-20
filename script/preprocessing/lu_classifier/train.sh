#!/bin/bash

source_dir=./source/preprocessing/lu_classifier
data_dir=./data/preprocessing/lu_classifier

pretrained_model=bert-base-uncased
n_splits=5

# text_input_styles=(sep token0 token00)
# text_input_styles=(sep)
# text_input_styles=(token0)
text_input_styles=(token00)

modes=(train)


device=cuda:0
for mode in "${modes[@]}";do
    for text_input_style in "${text_input_styles[@]}";do
        for ((part=0; part<n_splits; part++));do
            uv run python ${source_dir}/lu_classifier.py \
                --part "${part}" \
                --n_splits "${n_splits}" \
                --pretrained_model "${pretrained_model}" \
                --input_dir "${data_dir}/dataset" \
                --output_model_dir "${data_dir}/models/${pretrained_model}/${text_input_style}/${mode}/${n_splits}_${part}" \
                --device ${device} \
                --mode "${mode}" \
                --text_input_style "${text_input_style}"
        done
    done
done