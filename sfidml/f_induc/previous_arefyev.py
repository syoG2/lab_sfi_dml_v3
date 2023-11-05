from collections import Counter

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastcluster import linkage
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertForMaskedLM

from sfidml.modules.score_clustering import calculate_bcubed


class LemmatizationWithPOSTagger(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def pos_tag(self, tokens):
        pos_tokens = [nltk.pos_tag(token) for token in tokens]
        pos_tokens = [
            [
                (
                    word,
                    self.lemmatizer.lemmatize(
                        word, self._get_wordnet_pos(pos_tag)
                    ),
                    [pos_tag],
                )
                for (word, pos_tag) in pos
            ]
            for pos in pos_tokens
        ]
        return pos_tokens


def arefyev_collate_fn(batch):
    output_dict = {
        "verb": [],
        "frame": [],
        "ex_idx": [],
        "batch_size": len(batch),
    }
    for ita in ["input_ids", "token_type_ids", "attention_mask"]:
        output_dict[ita] = nn.utils.rnn.pad_sequence(
            [torch.LongTensor(b[ita]) for b in batch], batch_first=True
        )
    output_dict["target_tidx"] = torch.LongTensor(
        [b["target_tidx"] for b in batch]
    )

    for b in batch:
        output_dict["verb"].append(b["verb"])
        output_dict["frame"].append(b["frame"])
        output_dict["ex_idx"].append(b["ex_idx"])
    return output_dict


class ArefyevDataset(Dataset):
    def __init__(self, df, pretrained_model_name):
        self.df = df
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self.lem = LemmatizationWithPOSTagger()
        self._preprocess()

    def __len__(self):
        return len(self.out_inputs)

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.out_inputs = []
        for df_dict in self.df.to_dict("records"):
            text_widx_lem = [
                o[1]
                for o in self.lem.pos_tag([df_dict["text_widx"].split()])[0]
            ]
            inputs = self.tokenizer(text_widx_lem, is_split_into_words=True)

            target_tidx = inputs.word_ids().index(df_dict["target_widx"])
            inputs["target_tidx"] = target_tidx
            inputs.update(
                {
                    "frame": df_dict["frame"],
                    "verb": df_dict["verb"],
                    "ex_idx": df_dict["ex_idx"],
                }
            )
            self.out_inputs.append(inputs)


class ArefyevMlmDataset(Dataset):
    def __init__(self, df, pretrained_model_name):
        self.df = df
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self._preprocess()

    def __len__(self):
        return len(self.out_inputs)

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.out_inputs = []
        mlm_tokens = ["and", "then", self.tokenizer.mask_token]
        for df_dict in self.df.to_dict("records"):
            text_widx = df_dict["text_widx"].split()
            text_widx_mlm = (
                text_widx[: df_dict["target_widx"] + 1]
                + mlm_tokens
                + text_widx[df_dict["target_widx"] + 1 :]
            )
            inputs = self.tokenizer(text_widx_mlm, is_split_into_words=True)
            target_tidx = inputs.word_ids().index(
                df_dict["target_widx"] + len(mlm_tokens)
            )
            inputs["target_tidx"] = target_tidx
            inputs.update(
                {
                    "frame": df_dict["frame"],
                    "verb": df_dict["verb"],
                    "ex_idx": df_dict["ex_idx"],
                }
            )
            self.out_inputs.append(inputs)


class ArefyevEmbedding:
    def __init__(self, pretrained_model_name, batch_size, device):
        self.pretrained_model_name = pretrained_model_name
        self.batch_size = batch_size
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name, output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(
            pretrained_model_name, config=self.config
        )
        self.model = self.model.to(device).eval()
        self.mlm = BertForMaskedLM.from_pretrained(pretrained_model_name)
        self.mlm = self.mlm.to(device).eval()

        self.lem = LemmatizationWithPOSTagger()

        self.layer = 6
        self.topk = 40

    def get_embedding(self, df):
        ds = ArefyevDataset(df, self.pretrained_model_name)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=arefyev_collate_fn,
            shuffle=False,
        )

        vec_list = []
        for batch in dl:
            with torch.no_grad():
                outputs = self.model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["token_type_ids"].to(self.device),
                )["hidden_states"][self.layer]
                embs = outputs[
                    torch.LongTensor(range(len(batch["target_tidx"]))),
                    batch["target_tidx"],
                ]
                vec_list += list(embs.cpu().detach().numpy())

        df_vec = (
            df.reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "vec_id"})
        )
        vec_array = np.array(vec_list)
        return df_vec, vec_array

    def get_substitutions(self, df):
        dl = DataLoader(
            ArefyevMlmDataset(df, self.pretrained_model_name),
            batch_size=self.batch_size,
            collate_fn=arefyev_collate_fn,
            shuffle=False,
        )

        sub_list = []
        for batch in dl:
            with torch.no_grad():
                outputs = self.mlm(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    token_type_ids=batch["token_type_ids"].to(self.device),
                )
            preds = outputs[0][
                torch.LongTensor(range(len(batch["target_tidx"]))),
                batch["target_tidx"],
            ].topk(self.topk)
            subs = self.tokenizer.convert_ids_to_tokens(
                preds.indices.cpu().view(-1)
            )
            batch_sub_list = []
            for i in range(0, len(subs), self.topk):
                for sub in self.lem.pos_tag([subs[i : i + self.topk]]):
                    batch_sub_list.append(" ".join([s[1] for s in sub]))
            sub_list += batch_sub_list

        all_subs = sum([s.split() for s in sub_list], [])
        self.tfidf = TfidfVectorizer(
            max_features=len([w for w, _ in dict(Counter(all_subs)).items()])
        )

        df_tfidf = pd.DataFrame(
            self.tfidf.fit_transform(sub_list).toarray(),
            columns=["tfidf_" + n for n in self.tfidf.get_feature_names_out()],
        )
        return pd.merge(df, df_tfidf, left_index=True, right_index=True)


class ArefyevClustering:
    def __init__(self):
        self.method = "average"
        self.metric = "cosine"
        self.criterion1 = "distance"
        self.criterion2 = "maxclust"

        self.regex = "tfidf_"

        self.min_th = 20
        self.max_th = 25

    def _clustering_1st(self, vec_array, params):
        z = linkage(
            pdist(vec_array),
            method=self.method,
            metric=self.metric,
            preserve_input=False,
        )
        return fcluster(z, t=params["th"], criterion=self.criterion1)

    def _clustering_2nd(self, df):
        df2_list = []
        for frame_cluster_1st in tqdm(
            list(sorted(set(df["frame_cluster_1st"])))
        ):
            df_cluster = df[df["frame_cluster_1st"] == frame_cluster_1st].copy()
            if len(df_cluster) == 1:
                df_cluster["frame_cluster_2nd"] = [1]
            else:
                tfidf_array = df_cluster.filter(
                    regex=f"^({self.regex})", axis=1
                ).values
                z = linkage(
                    pdist(tfidf_array),
                    method=self.method,
                    metric=self.metric,
                    preserve_input=False,
                )
                df_cluster["frame_cluster_2nd"] = list(
                    fcluster(z, t=2, criterion=self.criterion2)
                )
            df2_list += df_cluster.to_dict("records")

        df2 = pd.DataFrame(df2_list)
        df2 = df2.drop(
            list(df2.filter(regex=self.regex, axis=1).columns), axis=1
        )
        df2["frame_cluster_1st_2nd"] = (
            df2["frame_cluster_1st"]
            .astype(str)
            .str.cat(df2["frame_cluster_2nd"].astype(str), sep="")
        )
        map_1to2 = {
            c: i + 1 for i, c in enumerate(set(df2["frame_cluster_1st_2nd"]))
        }
        return df2["frame_cluster_1st_2nd"].map(map_1to2)

    def make_params(self, df, vec_array):
        best_bcf = 0
        for th in tqdm(range(self.min_th, self.max_th + 1)):
            df["frame_cluster_1st"] = self._clustering_1st(
                vec_array, {"th": th}
            )
            df["frame_cluster"] = self._clustering_2nd(df)
            true = df.groupby("frame")["ex_idx"].agg(list).tolist()
            pred = df.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
            bcf = calculate_bcubed(true, pred)[2]
            if best_bcf <= bcf:
                best_bcf = bcf
                params = {"th": th}
        return params

    def step(self, df, vec_array, params):
        df["frame_cluster_1st"] = self._clustering_1st(vec_array, params)
        df["frame_cluster"] = self._clustering_2nd(df)
        df = df.filter(regex=f"^(?!{self.regex})", axis=1)
        return df
