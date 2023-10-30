import numpy as np
from fastcluster import linkage
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from sfidml.modules.score_clustering import calculate_bcubed


class AnwarEmbedding:
    def __init__(self, w2v_file):
        self.w2v = KeyedVectors.load_word2vec_format(w2v_file, binary=True)

    def get_embedding(self, df):
        self.w2v.init_sims(True)
        w2i = {w: i for i, w in enumerate(self.w2v.index_to_key)}

        tfidf = TfidfVectorizer(vocabulary=self.w2v.index_to_key)
        tfidf_array = tfidf.fit_transform(list(df["text_widx"]))
        norm_tfidf_array = normalize(tfidf_array, norm="l2", axis=1, copy=False)

        sum_tfidf = np.sum(norm_tfidf_array, axis=1)
        sum_tfidf[np.where(sum_tfidf == 0)] = 1
        context_vec = np.array(
            norm_tfidf_array.dot(self.w2v.vectors) / sum_tfidf
        )
        norm_context_vec = normalize(context_vec, norm="l2", axis=1)
        context_vec_dict = {
            ex_idx: norm_context_vec[idx]
            for idx, ex_idx in enumerate(df["ex_idx"])
        }

        vec_list = []
        for df_dict in df.to_dict("records"):
            text_widx = df_dict["text_widx"]
            target_widx = df_dict["target_widx"]
            if text_widx.split()[target_widx] in w2i:
                target_vec = self.w2v.get_vector(
                    text_widx.split()[target_widx], norm=False
                )
            else:
                target_vec = np.zeros(300)
            context_vec = context_vec_dict[df_dict["ex_idx"]]

            vec = np.concatenate([target_vec, context_vec], axis=0).reshape(
                -1, 1
            )
            vec_list.append(normalize(vec, norm="l2", axis=0).flatten())

        df_vec = (
            df.reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "vec_id"})
        )
        vec_array = np.array(vec_list)
        return df_vec, vec_array


class AnwarClustering:
    def __init__(self):
        self.method = "average"
        self.metric = "cityblock"
        self.criterion = "distance"
        self.th_min = 100
        self.th_max = 110

    def _clustering(self, vec_array, params):
        z = linkage(
            pdist(vec_array),
            method=self.method,
            metric=self.metric,
            preserve_input=False,
        )
        return fcluster(z, t=params["th"], criterion=self.criterion)

    def make_params(self, df, vec_array):
        z = linkage(
            pdist(vec_array),
            method=self.method,
            metric=self.metric,
            preserve_input=False,
        )
        best_bcf = 0
        for th in tqdm([i / 100 for i in range(self.th_min, self.th_max + 1)]):
            df["frame_cluster"] = fcluster(z, t=th, criterion=self.criterion)
            true = df.groupby("frame")["ex_idx"].agg(list).tolist()
            pred = df.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
            bcf = calculate_bcubed(true, pred)[2]
            if best_bcf <= bcf:
                best_bcf = bcf
                params = {"th": th}
        return params

    def step(self, df, vec_array, params):
        df["frame_cluster"] = self._clustering(vec_array, params)
        return df
