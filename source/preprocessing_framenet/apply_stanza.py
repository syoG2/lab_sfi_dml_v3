import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Union

import stanza
from fadml.utils.data_utils import read_json, write_json
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from tokenizations import get_alignments
from tqdm import tqdm


def make_alignment(
    text: Union[str, List[str]], new_text: Union[str, List[str]]
) -> Dict[int, int]:
    alignment = {}
    for idx, new_idx in enumerate(get_alignments(text, new_text)[0] + [[]]):
        if len(new_idx) != 0:
            alignment[idx] = new_idx[0]
        else:
            if idx - 1 in alignment:
                alignment[idx] = alignment[idx - 1]
            else:
                alignment[idx] = 0
    return alignment


def make_word_list(doc: Document) -> List[Dict[str, Any]]:
    word_list, count, word_count = [], 0, 0
    for sentence_id, sentence in enumerate(doc.sentences):
        child = {}
        for word in sentence.words:
            if word.head not in child:
                child[word.head] = [word.id]
            else:
                child[word.head].append(word.id)

        for word in sentence.words:
            word_dict = word.to_dict()
            word_dict.update(
                {
                    "id": count,
                    "sent_id": sentence_id,
                    "word_id": int(word.id) - 1,
                }
            )
            if word.head != 0:
                word_dict.update(
                    {
                        "head": word.head - 1 + word_count,
                        "head_text": sentence.words[word.head - 1].text,
                    }
                )
            else:
                word_dict.update({"head": -1, "head_text": "[ROOT]"})
            if word.id in child:
                word_dict.update(
                    {
                        "children": [
                            i - 1 + word_count for i in child[word.id]
                        ],
                    }
                )
            else:
                word_dict.update({"children": []})
            word_list.append(word_dict)
            count += 1
        word_count += len(sentence.words)
    return word_list


def find_head(word_list, span_start, span_end):
    cache = []
    children = [d["id"] for d in word_list if d["head"] == -1]
    for child_id in children:
        if child_id in cache:
            continue
        if span_start <= child_id <= span_end:
            return child_id
        else:
            children += word_list[child_id]["children"]
        cache.append(child_id)


def get_verb(lu_name: str, nlp: Pipeline) -> str:
    verb = re.sub("[\[|\(].+[\)|\]]", "", lu_name)[:-2].strip()
    if not re.fullmatch("[a-z][a-z-]*", verb):
        doc = nlp(verb)
        head = [
            w.id - 1
            for s in doc.sentences
            for w in s.words
            if w.deprel == "root"
        ][0]
        verb = [w.text for s in doc.sentences for w in s.words][head]
    return verb


def main(args: Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exemplars = read_json(args.input_exemplars_file)
    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=True,
        pos_batch_size=3000,
    )

    exemplars, word_lists = [], []
    for exemplar in tqdm(exemplars):
        if not exemplar["lu_name"].endswith(".v"):
            continue

        text_norm = " ".join(
            (re.sub("\s", " ", exemplar["text"]).rstrip() + " ").split() + [""]
        )
        doc = nlp(text_norm)
        text_widx = " ".join(
            [w.text for s in doc.sentences for w in s.words] + [""]
        )

        a1 = make_alignment(exemplar["text"], text_norm)
        a2 = make_alignment(text_norm, text_widx)
        a3 = make_alignment(list(text_widx), text_widx.split())
        target_widx = [a3[a2[a1[t]]] for t in exemplar["target"][0]]
        fe_widx = [
            [a3[a2[a1[b]]], a3[a2[a1[e]]], f] for b, e, f in exemplar["fe"][0]
        ]

        word_list = make_word_list(doc)
        target_widx_head = find_head(word_list, target_widx[0], target_widx[1])
        target_widx_with_head = target_widx + [target_widx_head]
        fe_widx_with_head = []
        for b, e, f in fe_widx:
            h = find_head(word_list, b, e)
            fe_widx_with_head.append([b, e, h, f])

        verb = get_verb(exemplar["lu_name"], nlp)

        exemplar.update(
            {
                "text_widx": text_widx,
                "target_widx": target_widx_with_head,
                "fe_widx": fe_widx_with_head,
                "verb": verb,
            }
        )
        exemplars.append(exemplar)
        word_lists.append(
            {"ex_idx": exemplar["ex_idx"], "word_list": word_list}
        )

    write_json(exemplars, args.output_dir / "exemplars.jsonl")
    write_json(word_lists, args.output_dir / "word_list.jsonl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_exemplars_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
