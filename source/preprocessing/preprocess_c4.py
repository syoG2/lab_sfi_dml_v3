import argparse
import re
from pathlib import Path
from unicodedata import normalize

import pandas as pd
import stanza
from base_data import BaseData, WordInfo, WordList
from collect_c4 import C4Id
from datasets import Dataset
from lu_classifier.util import (
    extract_entities,
    id2label,
    label2id,
    preprocess_data_sep,
    preprocess_data_token0,
    preprocess_data_token00,
    run_prediction,
)
from spacy_alignments import get_alignments
from timeout_decorator import TimeoutError, timeout
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

# tqdmをpandasのapplyメソッドで使用できるように設定
tqdm.pandas()


def get_pred_lu_name(text_widx, doc_sentence, pred_lu_idx):
    # 前処理後のtextのsplit()とdoc_sentence.wordsの対応を取る必要がある
    doc_words = [word.text for word in doc_sentence.words]
    char_to_word, _ = get_alignments(list(text_widx), doc_words)
    return (
        " ".join(
            [
                doc_sentence.words[i].lemma
                for idx in pred_lu_idx
                for i in range(
                    char_to_word[idx[0]][0], char_to_word[idx[-1] - 1][-1] + 1
                )
            ]
        )
        + ".v"
    )


def get_featured_word_idxs(text_widx, doc_sentence):
    doc_words = [word.text for word in doc_sentence.words]
    _, doc_to_char = get_alignments(list(text_widx), doc_words)
    return [
        [doc_to_char[word.id - 1][0], doc_to_char[word.id - 1][-1] + 1]
        for word in doc_sentence.words
        if word.upos == "VERB"
    ]


class C4Data(BaseData):
    id_data: C4Id  # 元データの参照に必要な情報を格納
    part_id: int
    target_widx: list[list[int]]
    target_widx_head: list[int]
    verb: str
    frame: str = ""  # フレーム情報を格納
    verb_frame: str  # verbとframeを結合した情報を格納


class C4WordList(WordList):
    id_data: C4Id
    # words: list[WordInfo]


def make_word_list(
    id_data: C4Id, doc_sentence: list[list], sequence_number: int
) -> WordList:
    # 構文解析の結果を整理して返す
    ret: C4WordList = C4WordList(id_data=id_data, words=[])
    root_id = -1
    for word in doc_sentence.words:
        try:
            word_info: WordInfo = WordInfo(
                id=len(ret.words),  # 複数sentence全体の連番に変更
                text=word.text,
                lemma=word.lemma,
                upos=word.upos,
                xpos=word.xpos,
                feats=word.feats,
                head=len(ret.words) + (word.head - word.id)
                if word.head != 0
                else -1,  # idの変更に合わせる
                deprel=word.deprel,
                start_char=word.start_char,
                end_char=word.end_char,
                children=[],
                word_idx=word.id - 1,
                sent_id=sequence_number,
                depth=0,
            )
            ret.words.append(word_info)
        except KeyError as e:
            print(f"key:'{e}'が存在しません。")
    for word_info in ret.words:
        if word_info.head != -1:
            ret.words[word_info.head].children.append(word_info.id)  # childrenを作成

    bfs = [root_id]
    while len(bfs) > 0:
        current = bfs.pop(0)
        for child in ret.words[current].children:
            ret.words[child].depth = ret.words[current].depth + 1
            bfs.append(child)

    return ret


@timeout(100)
def nlp_with_timeout(nlp, text):  # nlp(text)のタイムアウトを設定
    return nlp(text)


def get_doc_sentence(nlp: stanza.Pipeline, text: str) -> list:
    try:
        return [
            (seq_id, sentence)
            for seq_id, sentence in enumerate(nlp_with_timeout(nlp, text).sentences)
        ]
    except TimeoutError:
        return []
    except RuntimeError:
        return []


def lu_char_to_word_index(text: str, idx: list[list[int]]) -> list[int]:
    # luを文字単位から単語単位に変換
    char_to_word, _ = get_alignments(list(text), text.split() + [" "])
    return [
        [char_to_word[start_end[0]][0], char_to_word[start_end[-1] - 1][-1]]
        for start_end in idx
    ]


def get_target_widx_head(target_widx: list[list[int]], word_list: C4WordList):
    # targetのheadを取得
    if len(target_widx) == 0:
        return []

    ret = target_widx[0].copy()
    ret.append(target_widx[0][0])
    min_depth = word_list.words[target_widx[0][0]].depth
    for id in range(target_widx[0][0], target_widx[0][1]):
        if word_list.words[id].depth < min_depth:
            min_depth = word_list.words[id].depth
            ret[-1] = id

    for tw in target_widx:
        for id in range(tw[0], tw[1]):
            if word_list.words[id].depth < min_depth:
                min_depth = word_list.words[id].depth
                ret = tw
                ret.append(id)

    return ret


def get_verb_idx(doc: list[list]) -> int:
    # lu_nameをstanzaにかけたものからrootを取得
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.deprel == "root":
                return word.id - 1


def main(args):
    # outputディレクトリの作成
    args.output_exemplar_dir.mkdir(parents=True, exist_ok=True)
    args.output_wordlist_dir.mkdir(parents=True, exist_ok=True)

    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=True,
        device=args.device,
        pos_batch_size=3000,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path))
    model = AutoModelForTokenClassification.from_pretrained(str(args.model_path))
    model.to(args.device)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    df = pd.read_json(args.input_file, lines=True)
    df = df[args.part_id * 1000 : min((args.part_id + 1) * 1000, len(df))]

    tqdm.pandas(desc="normalize_text")
    df["normalize_text"] = df["text"].progress_apply(
        lambda x: normalize("NFKC", x)
    )  # Unicode正規化

    tqdm.pandas(desc="doc_sentence")
    df["doc_sentence"] = df["normalize_text"].progress_apply(
        lambda x: get_doc_sentence(nlp, x)
    )
    df = df[df["doc_sentence"].apply(len) != 0]

    df = df.explode("doc_sentence", True)
    # 連番をつける
    df["sequence_number"] = df["doc_sentence"].apply(lambda x: x[0])
    df["doc_sentence"] = df["doc_sentence"].apply(lambda x: x[1])

    df["text"] = df["doc_sentence"].apply(
        lambda x: x.text
    )  # 前処理前のtextを1文ごとにする

    tqdm.pandas(desc="text_widx")
    df["text_widx"] = df["doc_sentence"].progress_apply(
        lambda x: " ".join([word.text for word in x.words])
        + " "  # 末尾に空白を追加しないと[,,,"a"," "]と[,,,," ","a"]のアラインメントをとる時におかしくなる)
    )

    tqdm.pandas(desc="token_length")
    if args.text_input_style == "sep":
        df = df[
            df["text_widx"].progress_apply(
                lambda x: len(tokenizer(x)["input_ids"]) <= tokenizer.model_max_length
            )
        ]
        preprocess_data = preprocess_data_sep
    elif args.text_input_style == "token0":
        df = df[
            df["text_widx"].progress_apply(
                lambda x: len(tokenizer(x)["input_ids"]) + 1
                <= tokenizer.model_max_length
            )
        ]
        preprocess_data = preprocess_data_token0
    elif args.text_input_style == "token00":
        preprocess_data = preprocess_data_token00
        df = df[
            df["text_widx"].progress_apply(
                lambda x: len(tokenizer(x)["input_ids"]) + 2
                <= tokenizer.model_max_length
            )
        ]

    tqdm.pandas(desc="featured_word_idx")
    df["featured_word_idx"] = df.progress_apply(
        lambda row: get_featured_word_idxs(row["text_widx"], row["doc_sentence"]),
        axis=1,
    )
    df = df.explode("featured_word_idx", True)
    df = df.dropna(subset=["featured_word_idx"])

    tqdm.pandas(desc="featured_word")
    df["featured_word"] = df.progress_apply(
        lambda row: row["text_widx"][
            row["featured_word_idx"][0] : row["featured_word_idx"][-1]
        ],
        axis=1,
    )
    # df = df.dropna(subset=["featured_word"])
    # df["preprocessed_target_widx"] = [[0, 0] for _ in range(len(df))]

    dataset = Dataset.from_pandas(
        df[["featured_word", "text_widx", "featured_word_idx"]]
    )
    preprocessed_dataset = dataset.map(
        preprocess_data,
        fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "prediction": True},
        remove_columns=dataset.column_names,
    )
    dataloader = DataLoader(
        preprocessed_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=data_collator,
    )

    predictions = run_prediction(dataloader, model)

    results = extract_entities(predictions, dataset, tokenizer, id2label)
    df["pred_lu_idx"] = [result["pred_lu_idx"] for result in results]

    tqdm.pandas(desc="pred_lu_name")
    df["pred_lu_name"] = df.progress_apply(
        lambda row: get_pred_lu_name(
            row["text_widx"], row["doc_sentence"], row["pred_lu_idx"]
        ),
        axis=1,
    )
    df = df[df["pred_lu_name"].apply(lambda x: len(x) > 0)]

    df["pred_lu_name"] = df["pred_lu_name"].str.lower()

    preprocessed_word_lists: list[C4WordList] = [
        make_word_list(
            C4Id(**row["id_data"]), row["doc_sentence"], row["sequence_number"]
        )
        for _, row in df.iterrows()
    ]

    tqdm.pandas(desc="target_widx")
    df["target_widx"] = df.progress_apply(
        lambda row: lu_char_to_word_index(row["text_widx"], row["pred_lu_idx"]), axis=1
    )

    tqdm.pandas(desc="target_widx_head")
    df.reset_index(drop=True, inplace=True)
    df["target_widx_head"] = df.progress_apply(
        lambda row: get_target_widx_head(
            row["target_widx"], preprocessed_word_lists[row.name]
        ),
        axis=1,
    )

    tqdm.pandas(desc="verb_idx")
    df["verb_idx"] = df.progress_apply(
        lambda row: 0
        if " " not in row["pred_lu_name"]
        else get_verb_idx(
            nlp(re.sub(r"(\.v)|(\[.*?\])|(\(.*?\))", "", row["pred_lu_name"]).strip())
        ),
        axis=1,
    )

    tqdm.pandas(desc="verb")
    df["verb"] = df.progress_apply(
        lambda row: preprocessed_word_lists[row.name]
        .words[row["target_widx"][row["verb_idx"]][0]]
        .lemma
        if len(row["target_widx"]) > row["verb_idx"]
        and len(preprocessed_word_lists[row.name].words)
        > row["target_widx"][row["verb_idx"]][0]
        else "",
        axis=1,
    )
    df = df[df["verb"] != ""]
    df.sort_values(by="pred_lu_name", inplace=True)

    preprocessed_exemplars: list[C4Data] = [
        C4Data(
            source=row["source"],
            id_data=C4Id(**row["id_data"]),
            featured_word=row["featured_word"],
            featured_word_idx=row["featured_word_idx"],
            text=row["text"],
            text_widx=row["text_widx"],
            preprocessed_lu_idx=row["pred_lu_idx"],
            part_id=args.part_id,
            lu_name=row["pred_lu_name"],
            target_widx=row["target_widx"],
            target_widx_head=row["target_widx_head"],
            verb=row["verb"],
            frame="",
            verb_frame=f"{row['verb']}_",
        )
        for _, row in df.iterrows()
    ]

    with open(args.output_exemplar_dir / f"exemplar_{args.part_id}.jsonl", "w") as f:
        with tqdm(preprocessed_exemplars) as pbar:
            pbar.set_description("[write preprocessed_exemplars]")
            for exemplar in pbar:
                print(exemplar.model_dump_json(), file=f)

    # df = df.drop_duplicates(subset=["text_widx"])  # 重複を削除

    with open(args.output_wordlist_dir / f"word_list_{args.part_id}.jsonl", "w") as f:
        with tqdm(preprocessed_word_lists) as pbar:
            pbar.set_description("[write word_list]")
            for word_list in pbar:
                print(word_list.model_dump_json(), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--file_id", type=int, required=True)
    parser.add_argument("--part_id", type=int, required=True)
    parser.add_argument(
        "--split_name", type=str, choices=["train", "validation"], required=False
    )
    parser.add_argument("--output_exemplar_dir", type=Path, required=True)
    parser.add_argument("--output_wordlist_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--tokenizer_path", type=Path, required=True)
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
#     input_file: Path = Path("")
#     file_id: int = 0  # input_fileが指定されている場合は無視される
#     part_id: int  # 前処理に時間がかかるため、part_id*1000~(part_id+1)*1000行目のデータを処理する
#     split_name: str = "train"
#     output_exemplar_file: Path = Path("")
#     output_wordlist_file: Path = Path("")
#     device: str = "cuda:0"
#     model_name: str = "bert-base-uncased"
#     model_path: Path = Path("")
#     tokenizer_path: Path = Path("")

#     def model_post_init(self, __context):
#         output_dir: Path = Path(f"./datasets/c4/{self.split_name}_{self.file_id:05}")
#         if self.input_file == Path(""):
#             self.input_file = Path(f"./data/c4/{self.split_name}_{self.file_id:05}.jsonl")
#         if self.output_exemplar_file == Path(""):
#             self.output_exemplar_file = output_dir / Path(f"exemplars_{self.part_id}_token0.jsonl")
#         if self.output_wordlist_file == Path(""):
#             self.output_wordlist_file = output_dir / Path(f"word_list_{self.part_id}_token0.jsonl")
#         if self.model_path == Path(""):
#             self.model_path = Path(f"./src/make_datasets/lu_classifier_token0/models/{self.model_name}/best/42/5_0/best_model")
#         if self.tokenizer_path == Path(""):
#             self.tokenizer_path = Path(f"./src/make_datasets/lu_classifier_token0/models/{self.model_name}/best/42/5_0/tokenizer")
