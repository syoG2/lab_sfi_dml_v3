import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# import pandas as pd
import seaborn as sns

from sfidml.f_induc.embedding import read_embedding
from sfidml.modules.project_embedding import project_embedding
from sfidml.utils.data_utils import read_json


def save_visualization(df, output_path, title, detail=True):
    fig = plt.figure(figsize=(10, 10))
    fig.set_facecolor("white")
    if "-" in set(df["frame"]):
        sns.scatterplot(
            x="x",
            y="y",
            data=df[(df["frame"] == "-") & (df["source"] == "framenet")],
            color="lightgray",
            alpha=0.2,
            s=50,
            marker=".",
            edgecolor=None,
            legend=detail,
        )
    if "c4" in set(df["source"]):
        sns.scatterplot(
            x="x",
            y="y",
            data=df[df["source"] == "c4"],
            color="lightgray",
            alpha=0.2,
            s=50,
            marker="X",
            edgecolor=None,
            legend=detail,
        )
    grid = sns.scatterplot(
        x="x",
        y="y",
        data=df[(df["frame"] != "-") & (df["source"] == "framenet")],
        hue="frame",
        s=50,
        style="frame",
        # markers=["o", "X", "s", "P", "D", "^", "v", "*", "p", "h"],
        markers=["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
        edgecolor=None,
        legend=detail,
    )
    grid.set(xlabel="", ylabel="")

    if detail == True:
        handles, labels = grid.get_legend_handles_labels()
        plt.legend(handles, labels, fontsize=20)
        plt.title(title, fontsize=24)

    plt.tight_layout()
    plt.savefig(output_path / f"{title}.png")
    plt.close()


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # params = read_jsonl(args.input_params_file)
    params = read_json(args.input_params_file)
    alpha = params["alpha"]
    vec_type2run_number = params["vec_type2run_number"]
    df_vec, vec_array = read_embedding(
        args.input_dir, "test", vec_type2run_number, alpha
    )

    # df_frame = df_vec.groupby("frame").agg(set)
    df_frame = (
        df_vec[df_vec["source"] == "framenet"].groupby("frame")[["verb"]].agg(set)
    )
    df_frame["n_verbs"] = df_frame["verb"].apply(lambda x: len(x))
    frames = list(df_frame.sort_values("n_verbs", ascending=False)[:10].index)

    frame_dict = {}
    for f in sorted(set(df_vec["frame"])):
        frame_dict[f] = str(frames.index(f)) + ": " + f if f in frames else "-"
    df_vec["frame"] = df_vec["frame"].map(frame_dict)

    df_parts = df_vec.sort_values("frame")
    parts_array = vec_array[df_parts["vec_id"]]
    df_prj = project_embedding(df_parts, parts_array, args.random_state)

    file = f"{args.random_state}-w"
    save_visualization(df_prj, args.output_dir, file, True)
    file = f"{args.random_state}-wo"
    save_visualization(df_prj, args.output_dir, file, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--input_params_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--vec_type", type=str, choices=["word", "mask", "wm"])
    parser.add_argument("--random_state", type=int, default=2)
    args = parser.parse_args()
    print(args)
    main(args)
