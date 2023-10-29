import faiss


class SimilarityRanking:
    def __init__(self, ranking_method):
        self.ranking_method = ranking_method

    def ranking(self, df, vec_array):
        if self.ranking_method == "all_all":
            ranking_list = self._ranking_all(df, vec_array)
        else:
            ranking_list = self._ranking_select(df, vec_array)
        return ranking_list

    def _ranking_all(self, df, vec_array):
        n, d = vec_array.shape
        index = faiss.IndexFlatIP(d)
        index.add(vec_array)
        searched_indexes = index.search(vec_array, k=n)[1]

        frame2ex = {}
        for frame in sorted(set(df["frame"])):
            frame2ex[frame] = df[df["frame"] == frame]["ex_idx"].to_list()

        ranking_list = []
        for df_dict, searched_index in zip(
            df.to_dict("records"), searched_indexes
        ):
            ex_idx = df_dict["ex_idx"]
            frame = df_dict["frame"]
            true_idx = [e for e in frame2ex[frame] if e != ex_idx]
            if len(true_idx) == 0:
                continue
            pred_index = searched_index[1 : len(true_idx) + 1]
            pred_idx = df.loc[pred_index, "ex_idx"].to_list()
            pred_label = df.loc[pred_index, "frame"].to_list()

            ranking_list.append(
                {
                    "query_idx": ex_idx,
                    "query_label": frame,
                    "true_idx": true_idx,
                    "pred_idx": pred_idx,
                    "pred_label": pred_label,
                }
            )
        return ranking_list

    def _ranking_select(self, df, vec_array, ranking):
        ranking_list = []
        for vf in sorted(set(df["verb_frame"])):
            verb = vf.split("_")[0]
            frame = "_".join(vf.split("_")[1:])

            df_query = df[df["verb_frame"] == vf].reset_index(drop=True)
            vec_query = vec_array[df_query["vec_id"]]

            pos, neg = ranking.split("_")
            if pos == "same":
                pos_pattern = (df["frame"] == frame) & (df["verb"] == verb)
            elif pos == "diff":
                pos_pattern = (df["frame"] == frame) & (df["verb"] != verb)

            if neg == "same":
                neg_pattern = (df["frame"] != frame) & (df["verb"] == verb)
            elif neg == "diff":
                neg_pattern = (df["frame"] != frame) & (df["verb"] != verb)

            if sum(pos_pattern) == 0 or sum(neg_pattern) == 0:
                continue

            df_value = df[pos_pattern | neg_pattern].reset_index(drop=True)
            vec_value = vec_array[df_value["vec_id"]]

            n, d = vec_array.shape
            index = faiss.IndexFlatIP(d)
            index.add(vec_value)
            searched_indexes = index.search(vec_query, k=n)[1]

            searched_frames = searched_indexes[df_query.index]
            df_frame = df_value[df_value["frame"] == frame]
            for ex_idx, searched_frame in zip(
                df_query["ex_idx"].to_list(), searched_frames
            ):
                true_idx = [
                    e for e in df_frame["ex_idx"].to_list() if e != ex_idx
                ]
                if len(true_idx) == 0:
                    continue

                pred_index = searched_frame[: len(true_idx)]
                pred_idx = df_value.loc[pred_index, "ex_idx"].to_list()
                pred_label = df_value.loc[pred_index, "frame"].to_list()

                ranking_list.append(
                    {
                        "query_idx": ex_idx,
                        "query_label": frame,
                        "true_idx": true_idx,
                        "pred_idx": pred_idx,
                        "pred_label": pred_label,
                    }
                )
        return ranking_list
