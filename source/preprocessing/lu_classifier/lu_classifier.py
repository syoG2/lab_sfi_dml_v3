import argparse
from glob import glob
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed
from util import (
    compute_scores,
    convert_results_to_labels,
    extract_entities,
    id2label,
    label2id,
    preprocess_data_sep,
    preprocess_data_token0,
    preprocess_data_token00,
    run_prediction,
)


class Score(BaseModel):
    lu_size: int  # LUの単語数
    acc: float
    correct: int
    size: int
    sep_acc: float
    sep_correct: int
    sep_size: int


def main(args):
    set_seed(args.seed)
    args.output_model_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train" or args.mode == "random":
        df_list = [
            pd.read_json(
                args.input_dir / "random" / f"{args.n_splits}_{i}.jsonl", lines=True
            )
            for i in range(args.n_splits)
        ]
    else:
        df_list = [
            pd.read_json(
                args.input_dir / "distinct" / f"{args.n_splits}_{i}.jsonl", lines=True
            )
            for i in range(args.n_splits)
        ]

    if args.mode == "train":
        # データセットの作成
        train_dataset = Dataset.from_pandas(
            pd.concat(
                [
                    df_list[(args.part + i) % args.n_splits]
                    for i in range(args.n_splits - 1)
                ],
                ignore_index=True,
            )
        )
        validation_dataset = Dataset.from_pandas(
            df_list[(args.part + args.n_splits - 1) % args.n_splits]
        )
        dataset = DatasetDict(
            {"train": train_dataset, "validation": validation_dataset}
        )
    elif args.mode == "distinct" or args.mode == "random":
        # データセットの作成
        train_dataset = Dataset.from_pandas(
            pd.concat(
                [
                    df_list[(args.part + i) % args.n_splits]
                    for i in range(args.n_splits - 2)
                ],
                ignore_index=True,
            )
        )
        test_dataset = Dataset.from_pandas(
            df_list[(args.part + args.n_splits - 2) % args.n_splits]
        )
        validation_dataset = Dataset.from_pandas(
            df_list[(args.part + args.n_splits - 1) % args.n_splits]
        )
        dataset = DatasetDict(
            {
                "train": train_dataset,
                "test": test_dataset,
                "validation": validation_dataset,
            }
        )

    model = AutoModelForTokenClassification.from_pretrained(
        args.pretrained_model, label2id=label2id, id2label=id2label
    )
    # トークナイザーとモデルの準備
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if args.text_input_style == "token0" or args.text_input_style == "token00":
        # カスタムトークン[unused0]を追加
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [tokenizer.convert_ids_to_tokens(1)]}
        )
        # モデルのエンベディング層をリサイズ
        model.resize_token_embeddings(len(tokenizer))

    for param in model.parameters():
        param.data = param.data.contiguous()

    data_collator = DataCollatorForTokenClassification(tokenizer)

    if args.text_input_style == "sep":
        preprocess_data = preprocess_data_sep
    elif args.text_input_style == "token0":
        preprocess_data = preprocess_data_token0
    elif args.text_input_style == "token00":
        preprocess_data = preprocess_data_token00

    # 訓練セットに対して前処理を行う
    train_dataset = dataset["train"].map(
        preprocess_data,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
        },
        remove_columns=dataset["train"].column_names,
    )

    # 検証セットに対して前処理を行う
    validation_dataset = dataset["validation"].map(
        preprocess_data,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
        },
        remove_columns=dataset["validation"].column_names,
    )

    # Trainerに渡す引数を初期化する
    training_args = TrainingArguments(
        output_dir=args.output_model_dir,  # 結果の保存フォルダ
        per_device_train_batch_size=32,  # 訓練時のバッチサイズ
        per_device_eval_batch_size=32,  # 評価時のバッチサイズ
        learning_rate=1e-4,  # 学習率
        lr_scheduler_type="linear",  # 学習率スケジューラ
        warmup_ratio=0.1,  # 学習率のウォームアップ
        num_train_epochs=5,  # 訓練エポック数
        evaluation_strategy="epoch",  # 評価タイミング
        save_strategy="epoch",  # チェックポイントの保存タイミング
        logging_strategy="epoch",  # ロギングのタイミング
        fp16=True,  # 自動混合精度演算の有効化
        report_to="none",  # 外部ツールへのログを無効化
    )

    # Trainerを初期化する
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    # 訓練する
    trainer.train()

    # ミニバッチの作成にDataLoaderを用いる
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=data_collator,
    )

    best_score = 0
    # 各チェックポイントで処理する
    for checkpoint in sorted(glob(str(args.output_model_dir / "checkpoint-*"))):
        # モデルを読み込む
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        model.to(args.device)  # モデルをGPUに移動
        # 固有表現ラベルを予測する
        predictions = run_prediction(validation_dataloader, model)

        # 固有表現を抽出する
        results = extract_entities(
            predictions, dataset["validation"], tokenizer, id2label
        )

        # 正解データと予測データのラベルのlistを作成する
        true_labels, pred_labels = convert_results_to_labels(results)
        # 評価スコアを算出する
        scores = compute_scores(true_labels, pred_labels, "micro")
        if best_score < scores["f1-score"]:
            best_score = scores["f1-score"]
            best_model = model

    # トークナイザーとモデルを保存する
    tokenizer.save_pretrained(str(args.output_model_dir / "tokenizer"))
    best_model.save_pretrained(str(args.output_model_dir / "best_model"))

    # 保存したトークナイザーとモデルを読み込む
    tokenizer = AutoTokenizer.from_pretrained(args.output_model_dir / "tokenizer")
    best_model = AutoModelForTokenClassification.from_pretrained(
        args.output_model_dir / "best_model"
    )

    if args.mode == "distinct" or args.mode == "random":
        test_dataset = dataset["test"].map(
            preprocess_data,
            fn_kwargs={
                "tokenizer": tokenizer,
                "label2id": label2id,
            },
            remove_columns=dataset["test"].column_names,
        )
        # ミニバッチの作成にDataLoaderを用いる
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=data_collator,
        )
        # 固有表現ラベルを予測する
        predictions = run_prediction(test_dataloader, best_model)

        # 固有表現を抽出する
        results = extract_entities(predictions, dataset["test"], tokenizer, id2label)

        # LUの単語数ごとの正解率を計算する
        lu_len_scores = {}
        for result in results:
            true_length = len(result["preprocessed_lu_idx"])
            if true_length not in lu_len_scores:
                lu_len_scores[true_length] = {
                    "size": 0,
                    "correct": 0,
                    "sep_correct": 0,
                    "sep_size": 0,
                }

            lu_len_scores[true_length]["size"] += 1
            if result["preprocessed_lu_idx"] == result["pred_lu_idx"]:
                lu_len_scores[true_length]["correct"] += 1

            is_sep = False
            for i in range(len(result["preprocessed_lu_idx"]) - 1):
                if (
                    result["preprocessed_lu_idx"][i][-1] + 1
                    != result["preprocessed_lu_idx"][i + 1][0]
                ):
                    is_sep = True
                    break

            if is_sep:
                lu_len_scores[true_length]["sep_size"] += 1
                if result["preprocessed_lu_idx"] == result["pred_lu_idx"]:
                    lu_len_scores[true_length]["sep_correct"] += 1

        scores: list[Score] = [
            Score(
                lu_size=key,
                acc=value["correct"] / value["size"],
                correct=value["correct"],
                size=value["size"],
                sep_acc=value["sep_correct"] / value["sep_size"]
                if value["sep_size"] != 0
                else -1,
                sep_correct=value["sep_correct"],
                sep_size=value["sep_size"],
            )
            for key, value in sorted(lu_len_scores.items(), key=lambda x: x[0])
        ]

        with open(args.output_model_dir / "score.jsonl", "w") as f:
            for score in scores:
                print(score.model_dump_json(), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part", type=int, required=True
    )  # 分割交差検証用のパラメータ(0~n_splits-1)
    parser.add_argument("--n_splits", type=int, default=5)  # 分割数
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_model_dir", type=Path, required=True)
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["distinct", "random", "train"],
        required=False,
    )
    parser.add_argument(
        "--text_input_style",
        type=str,
        choices=["sep", "token0", "token00"],
        required=False,
    )
    args = parser.parse_args()
    print(args)
    main(args)

# class Args(BaseModel):
#     part: int  # 分割交差検証用のパラメータ(0~n_splits-1)
#     n_splits: int = 5  # 分割数
#     input_file: Path = Path("./datasets/framenet/exemplars.jsonl")
#     output_model_dir: Path = Path("")
#     pretrained_model: str = "bert-base-uncased"
#     device: str = "cuda:0"
#     seed: int = 42

#     def model_post_init(self, __context):
#         wd = Path(__file__)
#         if self.output_model_dir == Path(""):
#             self.output_model_dir = wd.parent / f"models/{self.pretrained_model}/lu/{self.n_splits}_{self.part}"
