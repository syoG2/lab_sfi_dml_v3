import argparse
from pathlib import Path

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from sfidml.f_induc.embedding import read_embedding
from sfidml.utils.data_utils import read_json, read_jsonl, write_jsonl
from sfidml.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_params_file is not None:
        best_params = read_json(args.input_params_file)
        for key, value in best_params.items():
            setattr(args, key, value)
    else:
        if args.alpha == 0:
            args.vec_type2run_number = {"word": args.run_number}
        elif args.alpha == 1:
            args.vec_type2run_number = {"mask": args.run_number}
        else:
            run_number_word, run_number_mask = args.run_number.split("-")
            args.vec_type2run_number = {
                "word": run_number_word,
                "mask": run_number_mask,
            }

    c4_cluster = pd.DataFrame(
        read_jsonl(args.input_clustering_dir / "exemplars_test-c4.jsonl")
    )

    # [ ]:plu_local,plu_globalを無視させる方法を考える
    if ("plu_local" in c4_cluster.keys()) and ("plu_global" in c4_cluster.keys()):
        c4_cluster = c4_cluster[["ex_idx", "plu_local", "plu_global", "frame_cluster"]]
    else:
        c4_cluster = c4_cluster[["ex_idx", "frame_cluster"]]

    df_vec_c4, vec_array_c4 = read_embedding(
        args.input_embedding_dir, "test-c4", args.vec_type2run_number, args.alpha
    )
    # df_vec_c4 = pd.merge(df_vec_c4, c4_cluster, on="ex_idx", how="outer")
    df_vec_c4 = pd.merge(df_vec_c4, c4_cluster, on="ex_idx")
    df_vec_framenet, vec_array_framenet = read_embedding(
        args.input_embedding_dir, "test-framenet", args.vec_type2run_number, args.alpha
    )

    classifier = KNeighborsClassifier(metric="euclidean", n_neighbors=1)

    classifier.fit(vec_array_c4, df_vec_c4["frame_cluster"])
    predict = classifier.predict(vec_array_framenet)
    df_vec_framenet["frame_cluster"] = predict

    # [ ]:plu_local,plu_globalを無視させる方法を考える
    # if ("plu_local" in c4_cluster.keys()) and ("plu_global" in c4_cluster.keys()):
    df_vec_framenet["plu_local"] = -1
    df_vec_framenet["plu_global"] = -1

    write_jsonl(
        df_vec_framenet.to_dict("records"),
        (args.output_dir / "exemplars_test-framenet.jsonl"),
    )

    df_vec = pd.concat([df_vec_c4, df_vec_framenet], join="inner")
    write_jsonl(
        df_vec.to_dict("records"),
        args.output_dir / "exemplars_test.jsonl",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_embedding_dir", type=Path, required=True)
    parser.add_argument("--input_clustering_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--input_params_file", type=Path, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
