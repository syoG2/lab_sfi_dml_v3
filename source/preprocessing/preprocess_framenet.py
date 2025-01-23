import argparse
import re
from pathlib import Path
from unicodedata import normalize

import pandas as pd
from base_data import BaseData, WordInfo, WordList
from collect_framenet import FramenetId
from spacy_alignments import get_alignments
from stanza.pipeline.core import Pipeline
from tqdm import tqdm

# tqdmをpandasのapplyメソッドで使用できるように設定
tqdm.pandas()


class FramenetData(BaseData):
    # 前処理後のframenet
    id_data: FramenetId  # 元データの参照に必要な情報を格納
    target: list[list[int]]  # 前処理前のLUの位置(文字レベル）
    fe: list[list[list[int | str]] | dict]  # 前処理前のfeの位置(文字レベル)
    target_widx: list[list[int]]
    fe_widx: list[list[list[int | str]] | dict[str, str]]
    target_widx_head: list[int]
    fe_widx_head: list[list[list[int | str]] | dict[str, str]]
    verb: str
    verb_frame: str
    frame_name: str
    frame_id: int
    lu_id: int
    fe_idx: list[
        list[list[int | str]] | dict[str, str]
    ]  # 前処理後のfeの位置(文字レベル)[[開始位置,終了位置],{省略されているfe名,省略の種類}]


class FramenetWordList(WordList):
    id_data: FramenetId
    # words: list[WordInfo]


def get_lu_idx(text: str, text_widx: str, target: list[list[int]]) -> list[list[int]]:
    # 前処理前後でtargetの位置を揃える(文字レベル)
    text_to_text_widx, _ = get_alignments(list(text), list(text_widx))
    return [
        [text_to_text_widx[t[0]][0], text_to_text_widx[t[1] - 1][-1] + 1]
        for t in target
    ]


def get_fe_idx(
    text: str, text_widx: str, fe: list[list[list[int | str]] | dict]
) -> list[list[int]]:
    # 前処理前後でfeの位置を揃える(文字レベル)
    text_to_text_widx, text_widx_to_text = get_alignments(list(text), list(text_widx))
    ret: list[list[list[int | str]] | dict] = [[], {}]
    ret[0] = [
        [text_to_text_widx[f[0]][0], text_to_text_widx[f[1] - 1][-1] + 1, f[2]]
        for f in fe[0]
    ]
    ret[1] = fe[1]
    return ret


def make_word_list(id_data: FramenetId, doc: list[list]) -> FramenetWordList:
    # 構文解析の結果を整理して返す
    ret: FramenetWordList = FramenetWordList(id_data=id_data, words=[])
    root_id = -1
    for sent_id, sentence in enumerate(doc.sentences):
        for word in sentence.words:
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
                    sent_id=sent_id,
                    depth=0,
                )
                ret.words.append(word_info)
                if word_info.deprel == "root":
                    root_id = word_info.id
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


def get_verb_idx(doc: list[list]) -> int:
    # lu_nameをstanzaにかけたものからrootを取得
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.deprel == "root":
                return word.id - 1


def lu_char_to_word_index(text: str, idx: list[list[int]]) -> list[int]:
    # luを文字単位から単語単位に変換
    char_to_word, _ = get_alignments(list(text), text.split() + [" "])
    return [
        [char_to_word[start_end[0]][0], char_to_word[start_end[-1] - 1][-1]]
        for start_end in idx
    ]


def fe_char_to_word_index(text: str, fe_idx: list[list[int]]) -> list[list[int]]:
    # feを文字単位から単語単位に変換
    char_to_word, _ = get_alignments(list(text), text.split() + [" "])
    ret = [
        [
            [char_to_word[fe[0]][0], char_to_word[fe[1] - 1][-1], fe[2]]
            for fe in fe_idx[0]
        ]
    ]
    ret.append(fe_idx[1])
    return ret


def get_target_widx_head(target_widx: list[list[int]], word_list: FramenetWordList):
    # targetのheadを取得
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


def get_fe_widx_head(
    fe_widx: list[list[list[int | str]] | dict], word_list: FramenetWordList
):
    ret = fe_widx.copy()
    for r in ret[0]:
        r.append(r[0])
        min_depth = word_list.words[r[0]].depth
        for id in range(r[0], r[1]):
            if word_list.words[id].depth < min_depth:
                min_depth = word_list.words[id].depth
                r[-1] = id
    return ret


def make_verb(lu_name, nlp):
    verb = re.sub("[\[|\(].+[\)|\]]", "", lu_name)[:-2].strip()
    if not re.fullmatch("[a-z][a-z-]*", verb):
        doc = nlp(verb)
        head = [
            word.id - 1
            for sentences in doc.sentences
            for word in sentences.words
            if word.deprel == "root"
        ][0]
        verb = [word.text for sentences in doc.sentences for word in sentences.words][
            head
        ]
    return verb


def main(args):
    # outputディレクトリの作成
    args.output_exemplar_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_wordlist_file.parent.mkdir(parents=True, exist_ok=True)

    nlp = Pipeline(
        "en",
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=True,
        device=args.device,
        pos_batch_size=9000,
    )

    # 前処理前のFrameNetのデータを読み込む
    df = pd.read_json(args.input_file, lines=True)

    df = df[df["lu_name"].str.contains(r"\.v")]  # 動詞のLUを抽出
    df = df[~df["lu_name"].str.contains(r"[\(\)\[\]]")]  # ()や[]が入っている例を除外
    df = df[
        df["lu_name"].apply(
            lambda lu_name: len(
                re.sub(r"(\.v)|(\[.*?\])|(\(.*?\))", "", lu_name).split()
            )
        )
        == df["target"].apply(lambda target: len(target))
    ]  # LUの単語数とtargetの単語数が一致するものを抽出(アノテーションミスと見られるものを省く)

    tqdm.pandas(desc="doc")
    df["doc"] = df["text"].progress_apply(
        lambda x: nlp(normalize("NFKC", x))
    )  # Unicode正規化 + 構文解析

    word_lists: list[FramenetWordList] = [
        make_word_list(FramenetId(id=dictionary["id_data"]["id"]), dictionary["doc"])
        for dictionary in df[["doc", "id_data"]].to_dict(orient="records")
    ]

    tqdm.pandas(desc="text_widx")
    df["text_widx"] = df["doc"].progress_apply(
        lambda x: " ".join([word.text for sent in x.sentences for word in sent.words])
        + " "  # 末尾に空白を追加しないと[,,,"a"," "]と[,,,," ","a"]のアラインメントをとる時におかしくなる
    )

    tqdm.pandas(desc="preprocessed_lu_idx")
    # 前処理後のLUの位置を格納(文字レベル)
    df["preprocessed_lu_idx"] = df.progress_apply(
        lambda row: get_lu_idx(row["text"], row["text_widx"], row["target"]), axis=1
    )

    tqdm.pandas(desc="target_widx")
    # 前処理後のLUの位置を格納(単語レベル)
    df["target_widx"] = df.progress_apply(
        lambda row: lu_char_to_word_index(row["text_widx"], row["preprocessed_lu_idx"]),
        axis=1,
    )

    tqdm.pandas(desc="target_widx_head")
    # 前処理後のLUの内、構文木上で最も根に近い単語の位置を取得
    df.reset_index(drop=True, inplace=True)
    df["target_widx_head"] = df.progress_apply(
        lambda row: get_target_widx_head(row["target_widx"], word_lists[row.name]),
        axis=1,
    )

    tqdm.pandas(desc="fe_idx")
    df["fe_idx"] = df.progress_apply(
        lambda row: get_fe_idx(row["text"], row["text_widx"], row["fe"]), axis=1
    )  # feの位置を単語単位に変換

    tqdm.pandas(desc="fe_widx")
    df["fe_widx"] = df.progress_apply(
        lambda row: fe_char_to_word_index(row["text_widx"], row["fe_idx"]), axis=1
    )

    tqdm.pandas(desc="fe_widx_head")
    df.reset_index(drop=True, inplace=True)
    df["fe_widx_head"] = df.progress_apply(
        lambda row: get_fe_widx_head(row["fe_widx"], word_lists[row.name]), axis=1
    )

    tqdm.pandas(desc="verb_idx")
    df["verb_idx"] = df.progress_apply(
        lambda row: 0
        if " " not in row["lu_name"]
        else get_verb_idx(
            nlp(re.sub(r"(\.v)|(\[.*?\])|(\(.*?\))", "", row["lu_name"]).strip())
        ),
        axis=1,
    )

    tqdm.pandas(desc="verb")
    if args.verb_form == "original":
        df["verb"] = df.progress_apply(
            lambda row: make_verb(row["lu_name"], nlp), axis=1
        )
    elif args.verb_form == "lemma":
        df["verb"] = df.progress_apply(
            lambda row: word_lists[row.name]
            .words[row["target_widx"][row["verb_idx"]][0]]
            .lemma,
            axis=1,
        )

    tqdm.pandas(desc="featured_word_idx")
    df["featured_word_idx"] = df.progress_apply(
        lambda row: row["preprocessed_lu_idx"][row["verb_idx"]],
        axis=1,
    )  # 注目する単語(動詞)の位置を取得

    tqdm.pandas(desc="featured_word")
    df["featured_word"] = df.progress_apply(
        lambda x: x["text_widx"][x["featured_word_idx"][0] : x["featured_word_idx"][1]],
        axis=1,
    )  # 注目する単語(動詞)を取得

    df["check_duplicates"] = df["target_widx_head"].apply(lambda x: x[2])
    df = df.drop_duplicates(subset=["text_widx", "check_duplicates"])  # 重複を削除

    preprocessed_exemplars: list[FramenetData] = [
        FramenetData(
            source=row["source"],
            id_data=FramenetId(id=row["id_data"]["id"]),
            text=row["text"],
            target=row["target"],
            fe=row["fe"],
            featured_word=row["featured_word"],
            featured_word_idx=row["featured_word_idx"],
            text_widx=row["text_widx"],
            preprocessed_lu_idx=row["preprocessed_lu_idx"],
            frame_name=row["frame_name"],
            frame_id=row["frame_id"],
            lu_name=row["lu_name"],
            lu_id=row["lu_id"],
            fe_idx=row["fe_idx"],
            ex_idx=row["ex_idx"],
            target_widx=row["target_widx"],
            fe_widx=row["fe_widx"],
            target_widx_head=row["target_widx_head"],
            fe_widx_head=row["fe_widx_head"],
            verb=row["verb"],
            verb_frame=f"{row['verb']}_{row['frame_name']}",
        )
        for _, row in df.iterrows()
    ]

    with open(args.output_exemplar_file, "w") as f:
        with tqdm(preprocessed_exemplars) as pbar:
            pbar.set_description("[write preprocessed_exemplars]")
            for exemplar in pbar:
                print(exemplar.model_dump_json(), file=f)

    with open(args.output_wordlist_file, "w") as f:
        with tqdm(word_lists) as pbar:
            pbar.set_description("[write word_list]")
            for word_list in pbar:
                print(word_list.model_dump_json(), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("./data/preprocessing/framenet/collect/exemplars.jsonl"),
    )
    parser.add_argument(
        "--output_exemplar_file",
        type=Path,
        default=Path("./data/preprocessing/framenet/preprocessing/exemplars.jsonl"),
    )
    parser.add_argument(
        "--output_wordlist_file",
        type=Path,
        default=Path("./data/preprocessing/framenet/preprocessing/word_list.jsonl"),
    )
    parser.add_argument(
        "--verb_form", type=str, choices=["lemma", "original"], default="original"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    print(args)
    main(args)


# class Args(BaseModel):
#     input_file: Path = Path("./data/preprocessing_framenet/collect/exemplars.jsonl")
#     output_exemplar_file: Path = Path("./data/preprocessing_framenet/preprocessing/exemplars.jsonl")
#     output_wordlist_file: Path = Path("./data/preprocessing_framenet/preprocessing/word_list.jsonl")
#     device: str = "cuda:0"
