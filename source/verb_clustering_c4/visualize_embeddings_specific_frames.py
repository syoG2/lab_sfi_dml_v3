import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sfidml.f_induc.embedding import read_embedding
from sfidml.modules.project_embedding import project_embedding
from sfidml.utils.data_utils import read_jsonl


def save_visualization(df, output_path, title, detail=True):
    fig = plt.figure(figsize=(10, 10))
    fig.set_facecolor("white")
    f_list = sorted(set(df["frame_name"]))
    c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    print(f_list, c_list)
    color_dict = {k: v for k, v in zip(f_list, c_list)}
    print(set(df["frame_name"]))
    grid = sns.scatterplot(
        x="x",
        y="y",
        data=df[::-1],
        hue="frame_name",
        palette=color_dict,
        style="frame_name",
        markers=["P", "s", "X", "o"],
        s=120,
        edgecolor=None,
        legend=None,
    )
    grid.set(xlabel="", ylabel="")

    handles, labels = grid.get_legend_handles_labels()
    plt.legend(handles, labels, fontsize=24)

    if detail == True:
        id_list = [
            25391,
            25396,
            12173,
            12016,
            18031,
            18155,
            19637,
            19564,
            17939,
            19465,
        ]
        for x, y, i, v in zip(df["x"], df["y"], df["vec_id"], df["verb"]):
            if i in id_list:
                plt.text(
                    x,
                    y,
                    v + "\n" + str(i),
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=12,
                )

    plt.tight_layout()
    plt.savefig(output_path + title + ".png")
    plt.close()


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    params = read_jsonl(args.input_params_file)
    alpha = params["alpha"]
    vec_type2run_number = params["vec_type2run_number"]
    df_vec, vec_array = read_embedding(
        args.input_dir, "test", vec_type2run_number, alpha
    )

    df_parts = df_vec[df_vec["frame"].isin(args.frames)].sort_values("frame")
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

    parser.add_argument("--frames", type=str, nargs="*")
    parser.add_argument("--random_state", type=int, default=0)
    args = parser.parse_args()
    print(args)
    main(args)
