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
# add_method=sequential_n_verb

# verb_form: 2段階クラスタリングにおける、1段階目の区分
#   originalでは動詞の原型、lemmaではstanzaにおけるlemmaをもとに区分
# verb_form=lemma
verb_form=original

# add_method: C4のデータの追加方法
#   frequency_100: 用例の最大数を100として、現れた順に用例を追加する
#   ratio: 各用例の出現頻度に基づいて、用例を追加する
add_method=frequency_100
add_method=ratio

# add_key: FrameNetの用例の何を基準にしてC4のデータを追加するか
#   lu_name: FrameNetの用例のlu_nameを基準にする
#   verb: FrameNetの用例の動詞を基準にする
# add_key=lu_name
add_key=verb

# clustering_dataset: クラスタリングの対象とするデータセット
#   c4first: C4のデータを最初にクラスタリングしてから、FrameNetのデータを割り当てる
#   mix: C4のデータとFrameNetのデータを混ぜてクラスタリングする
# clustering_dataset=c4first
# clustering_dataset=mix

# c4_rate: FrameNetの用例に対し、C4のデータを追加する割合
c4_rate=2

# c4_file_id: C4のデータのファイルID(0~c4_file_idまでのファイルを読み込む)
c4_file_id=6

uv run python ${source_dir}/make_dataset.py \
    --input_file ${input_dir}/${verb_form}/exemplars.jsonl \
    --output_dir ${data_dir}/${verb_form}/${add_method}/${add_key} \
    --setting_prefix ${setting_prefix} \
    --n_splits ${n_splits} \
    --c4_rate "${c4_rate}" \
    --add_method ${add_method} \
    --add_key ${add_key}\
    --c4_file_id ${c4_file_id}