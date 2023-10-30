import argparse
from pathlib import Path

import pandas as pd

from sfidml.modules.score_ranking import aggregate_ranking_scores
from sfidml.utils.data_utils import read_jsonl, write_json


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(args.input_score_file))

    df_train = pd.DataFrame(read_jsonl(args.input_train_file))
    train_frames = set(df_train["frame"])
    df_ol = df[df["query_label"].isin(train_frames)]
    df_nol = df[~df["query_label"].isin(train_frames)]

    ol_scores = aggregate_ranking_scores(df_ol.to_dict("records"))
    nol_scores = aggregate_ranking_scores(df_nol.to_dict("records"))

    metrics = {f"ol-{k}": v for k, v in ol_scores.items()}
    metrics.update({f"nol-{k}": v for k, v in nol_scores.items()})
    write_json(metrics, args.output_dir / f"metrics_{args.split}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_score_file", type=Path, required=True)
    parser.add_argument("--input_train_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
