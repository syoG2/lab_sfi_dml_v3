import argparse
from pathlib import Path

import pandas as pd


def main(args):
    for part in ["all_3_0", "all_3_1", "all_3_2"]:
        input_file = Path(
            f"./data/verb_clustering_c4/dataset/{part}/exemplars_test.jsonl"
        )
        output_file = Path(
            f"./data/verb_clustering_c4/dataset/{part}/exemplars_test.jsonl"
        )

        df = pd.read_json(input_file, lines=True)

        for lu_name, count in df["lu_name"].value_counts().items():
            df_framenet = df[(df["lu_name"] == lu_name) & (df["source"] == "framenet")]
            df_c4 = df[(df["lu_name"] == lu_name) & (df["source"] == "c4")]
            if len(df_framenet) > len(df_c4):
                df = df[df["lu_name"] != lu_name]

        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    print(args)
    main(args)
