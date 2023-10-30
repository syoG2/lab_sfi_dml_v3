import argparse
from pathlib import Path

import pandas as pd

from sfidml.f_induc.previous_anwar import AnwarClustering, AnwarEmbedding
from sfidml.utils.data_utils import read_jsonl, write_json, write_jsonl
from sfidml.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_dev = pd.DataFrame(read_jsonl(args.input_dev_file))
    df_test = pd.DataFrame(read_jsonl(args.input_test_file))

    embedding = AnwarEmbedding(args.input_w2v_file)
    df_vec_dev, vec_array_dev = embedding.get_embedding(df_dev)
    df_vec_test, vec_array_test = embedding.get_embedding(df_test)

    clustering = AnwarClustering()
    params = clustering.make_params(df_vec_dev, vec_array_dev)
    df_clu_dev = clustering.step(df_vec_dev, vec_array_dev, params)
    df_clu_test = clustering.step(df_vec_test, vec_array_test, params)

    write_jsonl(
        df_clu_dev.to_dict("records"),
        args.output_dir / "exemplars_dev.jsonl",
    )
    write_jsonl(
        df_clu_test.to_dict("records"),
        args.output_dir / "exemplars_test.jsonl",
    )
    write_json(vars(args), args.output_dir / "params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dev_file", type=Path, required=True)
    parser.add_argument("--input_test_file", type=Path, required=True)
    parser.add_argument("--input_w2v_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
