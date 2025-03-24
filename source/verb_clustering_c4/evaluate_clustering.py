from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

from sfidml.modules.score_clustering import calculate_clustering_scores
from sfidml.utils.data_utils import read_json, read_jsonl, write_json


def main(args: Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(args.input_file))
    df_c4 = df[df["source"] == "c4"]

    df_framenet = df[df["source"] == "framenet"]
    params = read_json(args.input_params_file)

    true = df_framenet.groupby("frame")["ex_idx"].agg(list).tolist()
    pred = df_framenet.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
    metrics = calculate_clustering_scores(true, pred)

    metrics.update(
        {
            "n_pred_clusters_c4": len(set(df_c4["frame_cluster"])),
        }
    )

    # [ ]:pluの評価の仕方を考える
    if params["clustering_name"] == "twostep":
        metrics.update(
            {
                "n_pred_lus_framenet": len(set(df_framenet["plu_global"])),
                "n_true_lus_framenet": len(set(df_framenet["verb_frame"])),
                "n_pred_lus_c4": len(set(df_c4["plu_global"])),
            }
        )
    elif params["clustering_name"] == "twostep_lu":
        # [ ]:LUを元に2段階クラスタリングした場合の評価値として適切か確認する
        metrics.update(
            {
                "n_pred_lus_framenet": len(set(df_framenet["plu_global"])),
                "n_true_lus_framenet": len(
                    set(
                        df_framenet.apply(
                            lambda row: row["lu_name"] + "_" + row["frame"], axis=1
                        )
                    )
                ),
                "n_pred_lus_c4": len(set(df_c4["plu_global"])),
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
