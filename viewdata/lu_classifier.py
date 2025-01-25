# LU抽出モデルのスコアを確認する
from pathlib import Path

import pandas as pd


def make_table(text_input_style, division_method):
    output_txt = ""
    df = pd.DataFrame(
        {
            "all": [],
            "1単語": [],
            "2単語": [],
            "3単語": [],
            "4単語": [],
            "5単語": [],
            "3単語以上": [],
        }
    )
    all = pd.DataFrame()
    output_txt += f"## {text_input_style} {division_method}\n"
    parts = ["5_0", "5_1", "5_2", "5_3", "5_4"]
    for part in parts:
        input_file = Path(
            f"./data/preprocessing/lu_classifier/models/bert-base-uncased/{text_input_style}/{division_method}/{part}/score.jsonl"
        )
        if not input_file.exists():
            output_txt += f"### {part} ファイルが存在しません\n"
            return output_txt
        tmp_df = pd.read_json(input_file, lines=True)
        tmp_df = tmp_df[["lu_size", "correct", "size"]]
        tmp_df["part"] = part
        all = pd.concat([all, tmp_df])

    df["all"] = all.groupby("part")[["correct", "size"]].apply(
        lambda x: f"{x['correct'].sum() / x['size'].sum():.4f} = {x['correct'].sum()}/{x['size'].sum()}"
    )
    df["1単語"] = (
        all[all["lu_size"] == 1]
        .groupby("part")[["correct", "size"]]
        .apply(
            lambda x: f"{x['correct'].sum() / x['size'].sum():.4f} = {x['correct'].sum()}/{x['size'].sum()}"
        )
    )
    df["2単語"] = (
        all[all["lu_size"] == 2]
        .groupby("part")[["correct", "size"]]
        .apply(
            lambda x: f"{x['correct'].sum() / x['size'].sum():.4f} = {x['correct'].sum()}/{x['size'].sum()}"
        )
    )
    df["3単語"] = (
        all[all["lu_size"] == 3]
        .groupby("part")[["correct", "size"]]
        .apply(
            lambda x: f"{x['correct'].sum() / x['size'].sum():.4f} = {x['correct'].sum()}/{x['size'].sum()}"
        )
    )
    df["4単語"] = (
        all[all["lu_size"] == 4]
        .groupby("part")[["correct", "size"]]
        .apply(
            lambda x: f"{x['correct'].sum() / x['size'].sum():.4f} = {x['correct'].sum()}/{x['size'].sum()}"
        )
    )
    df["5単語"] = (
        all[all["lu_size"] == 5]
        .groupby("part")[["correct", "size"]]
        .apply(
            lambda x: f"{x['correct'].sum() / x['size'].sum():.4f} = {x['correct'].sum()}/{x['size'].sum()}"
        )
    )
    df["3単語以上"] = (
        all[all["lu_size"] >= 3]
        .groupby("part")[["correct", "size"]]
        .apply(
            lambda x: f"{x['correct'].sum() / x['size'].sum():.4f} = {x['correct'].sum()}/{x['size'].sum()}"
        )
    )
    sum_df = all.groupby("lu_size").sum()
    wordall = f"{sum_df.loc[:, 'correct'].sum() / sum_df.loc[:, 'size'].sum():.4f} = {sum_df.loc[:, 'correct'].sum()}/{sum_df.loc[:, 'size'].sum()}"
    word1 = f"{sum_df.loc[1, 'correct'] / sum_df.loc[1, 'size']:.4f} = {sum_df.loc[1, 'correct']}/{sum_df.loc[1, 'size']}"
    word2 = f"{sum_df.loc[2, 'correct'] / sum_df.loc[2, 'size']:.4f} = {sum_df.loc[2, 'correct']}/{sum_df.loc[2, 'size']}"
    word3 = f"{sum_df.loc[3, 'correct'] / sum_df.loc[3, 'size']:.4f} = {sum_df.loc[3, 'correct']}/{sum_df.loc[3, 'size']}"
    word4 = f"{sum_df.loc[4, 'correct'] / sum_df.loc[4, 'size']:.4f} = {sum_df.loc[4, 'correct']}/{sum_df.loc[4, 'size']}"
    word5 = f"{sum_df.loc[5, 'correct'] / sum_df.loc[5, 'size']:.4f} = {sum_df.loc[5, 'correct']}/{sum_df.loc[5, 'size']}"
    word3ge = f"{sum_df.loc[3:, 'correct'].sum() / sum_df.loc[3:, 'size'].sum():.4f} = {sum_df.loc[3:, 'correct'].sum()}/{sum_df.loc[3:, 'size'].sum()}"
    row = pd.DataFrame(
        {
            "all": [wordall],
            "1単語": [word1],
            "2単語": [word2],
            "3単語": [word3],
            "4単語": [word4],
            "5単語": [word5],
            "3単語以上": [word3ge],
        },
        index=["micro平均"],
    )
    df = pd.concat([df, row])
    output_txt += df.to_markdown() + "\n"
    return output_txt


text_input_styles = ["sep", "token0", "token00"]
division_methods = ["distinct", "random"]
with open("./viewdata/lu_classifier.txt", "w") as f:
    for division_method in division_methods:
        for text_input_style in text_input_styles:
            txt = make_table(text_input_style, division_method)
            f.write(txt + "\n")
