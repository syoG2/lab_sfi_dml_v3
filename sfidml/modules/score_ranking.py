import numpy as np


def aggregate_ranking_scores(scores):
    return {
        "top1": np.array([score["top1"] for score in scores]).mean(),
        "acc": np.array([score["acc"] for score in scores]).mean(),
        "map": np.array([score["map"] for score in scores]).mean(),
    }


def calculate_ranking_scores(rankings):
    scores = []
    for r in rankings:
        if len(r["true_idx"]) != 0:
            r["top1"] = calculate_top1(r["query_label"], r["pred_label"])
            r["acc"] = calculate_accuracy(r["true_idx"], r["pred_idx"])
            r["map"] = calculate_map(r["query_label"], r["pred_label"])
            scores.append(r)
    return scores


def calculate_top1(query_label, pred_label):
    return 1 if query_label == pred_label[0] else 0


def calculate_accuracy(true_idx, pred_idx):
    return len(set(true_idx) & set(pred_idx)) / len(true_idx)


def calculate_map(query_label, pred_label):
    map, count = 0, 0
    for i, pl in enumerate(pred_label):
        if pl == query_label:
            count += 1
            map += count / (i + 1)
    return map / len(pred_label)
