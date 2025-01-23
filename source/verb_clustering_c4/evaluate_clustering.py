from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

from sfidml.modules.score_clustering import calculate_clustering_scores
from sfidml.utils.data_utils import read_json, read_jsonl, write_json


def main(args: Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(args.input_file))
    df = df[df["source"] == "framenet"]
    params = read_json(args.input_params_file)

    true = df.groupby("frame")["ex_idx"].agg(list).tolist()
    pred = df.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
    metrics = calculate_clustering_scores(true, pred)

    if params["clustering_name"] == "twostep":
        metrics.update(
            {
                "n_pred_lus": len(set(df["plu_global"])),
                "n_true_lus": len(set(df["verb_frame"])),
            }
        )
    elif params["clustering_name"] == "twostep_lu":
        # [ ]:LUを元に2段階クラスタリングした場合の評価値として適切か確認する
        metrics.update(
            {
                "n_pred_lus": len(set(df["plu_global"])),
                "n_true_lus": len(
                    set(
                        df.apply(
                            lambda row: row["lu_name"] + "_" + row["frame"], axis=1
                        )
                    )
                ),
            }
        )

    write_json(metrics, args.output_dir / f"metrics_{args.split}.json")
    write_json(params, args.output_dir / "params.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--input_params_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
