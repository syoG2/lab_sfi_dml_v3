import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sfidml.f_induc.embedding import BaseEmbedding
from sfidml.f_induc.model import BaseNet
from sfidml.utils.data_utils import read_jsonl, write_jsonl


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(args.input_file))
    params = list(read_jsonl(args.input_params_file))[0]

    model = BaseNet(
        params["pretrained_model_name"], params["normalization"], args.device
    )
    model.pretrained_model.load_state_dict(torch.load(args.input_model_file))
    model.to(args.device).eval()

    split = args.input_file.name.split(".jsonl")[0].split("_")[1]

    embedding = BaseEmbedding(
        model,
        params["pretrained_model_name"],
        params["vec_type"],
        args.batch_size,
    )
    df_vec, vec_array = embedding.get_embedding(df)
    write_jsonl(
        df_vec.to_dict("records"), args.output_dir / f"exemplars_{split}.jsonl"
    )

    vec_dict = {"vec": vec_array}
    np.savez_compressed(args.output_dir / f"vec_{split}", **vec_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--input_params_file", type=Path, required=True)
    parser.add_argument("--input_model_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    print(args)
    main(args)
