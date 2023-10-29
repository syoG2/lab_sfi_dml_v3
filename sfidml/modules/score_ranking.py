import numpy as np


def calc_ranking_scores(ranking_list):
    top1_list, acc_list = [], []
    for r in ranking_list:
        if len(r["true_idx"]) != 0:
            top1_list.append(calc_top1(r["query_label"], r["pred_label"]))
            acc_list.append(calc_accuracy(r["true_idx"], r["pred_idx"]))
    return {
        "top1": np.array(top1_list).mean(),
        "acc": np.array(acc_list).mean(),
    }


def calc_top1(query_label, pred_label):
    return 1 if query_label == pred_label[0] else 0


def calc_accuracy(true_idx, pred_idx):
    return len(set(true_idx) & set(pred_idx)) / len(true_idx)


def calc_map(query_label, pred_label):
    map, count = 0, 0
    for i, pl in enumerate(pred_label):
        if pl == query_label:
            count += 1
            map += count / (i + 1)
    return map / len(pred_label)
