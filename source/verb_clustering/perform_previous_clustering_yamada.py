import argparse
from pathlib import Path

import pandas as pd

from sfidml.f_induc.clustering_twostep import TwostepClustering
from sfidml.f_induc.embedding import BaseEmbedding
from sfidml.f_induc.model import BaseNet
from sfidml.utils.data_utils import read_jsonl, write_json, write_jsonl
from sfidml.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_dev = pd.DataFrame(read_jsonl(args.input_dev_file))
    df_test = pd.DataFrame(read_jsonl(args.input_test_file))

    pretrained_model_name = "bert-base-uncased"
    normalization = "false"
    model = BaseNet(pretrained_model_name, normalization, args.device).eval()

    embedding_word = BaseEmbedding(
        model, pretrained_model_name, "word", args.batch_size
    )
    embedding_mask = BaseEmbedding(
        model, pretrained_model_name, "mask", args.batch_size
    )

    df_vec_dev, word_array_dev = embedding_word.get_embedding(df_dev)
    _, mask_array_dev = embedding_mask.get_embedding(df_dev)
    df_vec_test, word_array_test = embedding_word.get_embedding(df_test)
    _, mask_array_test = embedding_mask.get_embedding(df_test)

    alpha = 0.7
    vec_array_dev = word_array_dev * (1 - alpha) + mask_array_dev * alpha
    vec_array_test = word_array_test * (1 - alpha) + mask_array_test * alpha

    clustering1 = "xmeans"
    clustering2 = "average"
    clustering = TwostepClustering(clustering1, clustering2)
    params = clustering.make_params(df_vec_dev, vec_array_dev)
    df_clu_dev = clustering.step(
        df_vec_dev, mask_array_dev, vec_array_dev, params
    )
    df_clu_test = clustering.step(
        df_vec_test, mask_array_test, vec_array_test, params
    )

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
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    print(args)
    main(args)
