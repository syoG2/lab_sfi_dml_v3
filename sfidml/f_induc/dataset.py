import random

from tokenizations import get_alignments
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def tokenize_text_and_target(tokenizer, text_widx, target_widx, vec_type):
    inputs = tokenizer(text_widx)
    text_tidx = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

    alignments, previous_char_idx_list = [], [1]
    for char_idx_list in get_alignments(text_widx.split(), text_tidx)[0]:
        if len(char_idx_list) == 0:
            alignments.append(previous_char_idx_list)
        else:
            alignments.append(char_idx_list)
            previous_char_idx_list = char_idx_list

    inputs["target_tidx"] = alignments[target_widx][0]
    if vec_type == "mask":
        inputs["input_ids"][inputs["target_tidx"]] = tokenizer.mask_token_id
    return inputs


class BaseDataset(Dataset):
    def __init__(self, df, pretrained_model_name, vec_type):
        self.df = df
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name
        )
        self.vec_type = vec_type
        self._preprocess()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.out_inputs = []
        for df_dict in self.df.to_dict("records"):
            inputs = tokenize_text_and_target(
                self.tokenizer,
                df_dict["text_widx"],
                df_dict["target_widx"],
                self.vec_type,
            )
            inputs.update(
                {
                    "frame": df_dict["frame"],
                    "verb": df_dict["verb"],
                    "ex_idx": df_dict["ex_idx"],
                }
            )
            self.out_inputs.append(inputs)
        self.data_num = len(self.out_inputs)


class SiameseDataset(Dataset):
    def __init__(self, df, pretrained_model_name, vec_type, vf2pos, vf2neg):
        self.df = df
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name
        )
        self.vec_type = vec_type
        self.vf2pos = vf2pos
        self.vf2neg = vf2neg
        self._preprocess()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.ex2inputs = {}
        for df_dict in self.df.to_dict("records"):
            inputs = tokenize_text_and_target(
                self.tokenizer,
                df_dict["text_widx"],
                df_dict["target_widx"],
                self.vec_type,
            )
            inputs.update(
                {
                    "frame": df_dict["frame"],
                    "verb": df_dict["verb"],
                    "ex_idx": df_dict["ex_idx"],
                }
            )
            self.ex2inputs[df_dict["ex_idx"]] = inputs

    def make_dataset(self):
        self.out_inputs = []
        for df_dict in self.df.to_dict("records"):
            true = random.choice([0, 1])
            if true == 1:
                candidates = [
                    e
                    for e in self.vf2pos[df_dict["verb_frame"]]
                    if e != df_dict["ex_idx"]
                ]
            else:
                candidates = self.vf2neg[df_dict["verb_frame"]]

            ex_idx = df_dict["ex_idx"]
            if len(candidates) != 0:
                ex_idx2 = random.choice(candidates)
            else:
                continue

            out_dict = {
                "instance1": self.ex2inputs[ex_idx],
                "instance2": self.ex2inputs[ex_idx2],
                "true": true,
            }
            out_dict["label"] = 1 if true == 1 else 0

            self.out_inputs.append(out_dict)
        self.data_num = len(self.out_inputs)


class TripletDataset(Dataset):
    def __init__(self, df, pretrained_model_name, vec_type, vf2pos, vf2neg):
        self.df = df
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name
        )
        self.vec_type = vec_type
        self.vf2pos = vf2pos
        self.vf2neg = vf2neg
        self._preprocess()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.ex2inputs = {}
        for df_dict in self.df.to_dict("records"):
            inputs = tokenize_text_and_target(
                self.tokenizer,
                df_dict["text_widx"],
                df_dict["target_widx"],
                self.vec_type,
            )
            inputs.update(
                {
                    "frame": df_dict["frame"],
                    "verb": df_dict["verb"],
                    "ex_idx": df_dict["ex_idx"],
                }
            )
            self.ex2inputs[df_dict["ex_idx"]] = inputs

    def make_dataset(self):
        self.out_inputs = []
        for df_dict in self.df.to_dict("records"):
            candidates_pos = [
                e
                for e in self.vf2pos[df_dict["verb_frame"]]
                if e != df_dict["ex_idx"]
            ]
            candidates_neg = self.vf2neg[df_dict["verb_frame"]]

            ex_idx = df_dict["ex_idx"]
            if (len(candidates_pos) != 0) and (len(candidates_neg) != 0):
                ex_pos = random.choice(candidates_pos)
                ex_neg = random.choice(candidates_neg)
            else:
                continue

            out_dict = {
                "anchor": self.ex2inputs[ex_idx],
                "positive": self.ex2inputs[ex_pos],
                "negative": self.ex2inputs[ex_neg],
            }
            self.out_inputs.append(out_dict)
        self.data_num = len(self.out_inputs)


class ClassificationDataset(Dataset):
    def __init__(self, df, pretrained_model_name, vec_type, frame2label=None):
        self.df = df
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name
        )
        self.vec_type = vec_type

        if frame2label is None:
            self.frame2label = self._make_frame2label()
        else:
            self.frame2label = frame2label
        self._preprocess()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.out_inputs = []
        for df_dict in self.df.to_dict("records"):
            inputs = tokenize_text_and_target(
                self.tokenizer,
                df_dict["text_widx"],
                df_dict["target_widx"],
                self.vec_type,
            )
            if df_dict["frame"] in self.frame2label:
                label = self.frame2label[df_dict["frame"]]
            else:
                label = self.frame2label["[OTHER]"]

            inputs.update(
                {
                    "frame": df_dict["frame"],
                    "verb": df_dict["verb"],
                    "ex_idx": df_dict["ex_idx"],
                    "label": label,
                }
            )
            self.out_inputs.append(inputs)
        self.data_num = len(self.out_inputs)

    def _make_frame2label(self):
        frame2label = {
            frame: idx
            for idx, frame in enumerate(
                ["[OTHER]"] + sorted(list(set(self.df["frame"])))
            )
        }
        return frame2label
