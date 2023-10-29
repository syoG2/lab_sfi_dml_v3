import itertools
from collections import defaultdict


def calculate_clustering_scores(true, pred):
    bcp, bcr, bcf = calculate_bcubed(true, pred)
    pu, ipu, puf = calculate_purity(true, pred)
    return {
        "n_pred_clusters": len(pred),
        "n_true_clusters": len(true),
        "n_pred_instances": len(sum(pred, [])),
        "n_true_instances": len(sum(true, [])),
        "bcp": bcp,
        "bcr": bcr,
        "bcf": bcf,
        "pu": pu,
        "ipu": ipu,
        "puf": puf,
    }


def calculate_bcubed(true, pred):
    true_map = defaultdict(set)
    true_map.update({e: set(c) for c in true for e in c})
    pred_map = defaultdict(set)
    pred_map.update({e: set(c) for c in pred for e in c})

    instances = set(itertools.chain(*true, *pred))
    sum_precision, sum_recall = 0, 0
    for instance in instances:
        n_commons = len(true_map[instance] & pred_map[instance])
        n_preds = len(pred_map[instance])
        n_trues = len(true_map[instance])
        if n_preds != 0:
            sum_precision += n_commons / n_preds
        if n_trues != 0:
            sum_recall += n_commons / n_trues

    avg_precision = sum_precision / len(instances)
    avg_recall = sum_recall / len(instances)
    f_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

    return avg_precision, avg_recall, f_score


def calculate_purity(true, pred):
    purity_sum = 0
    for pred_cluster in pred:
        max_true_intersection = 0
        for true_cluster in true:
            intersection = len(set(true_cluster) & set(pred_cluster))
            if intersection > max_true_intersection:
                max_true_intersection = intersection
        purity_sum += max_true_intersection
    purity = purity_sum / sum(len(c) for c in pred)

    inverse_purity_sum = 0
    for true_cluster in true:
        max_pred_intersection = 0
        for pred_cluster in pred:
            intersection = len(set(true_cluster) & set(pred_cluster))
            if intersection > max_pred_intersection:
                max_pred_intersection = intersection
        inverse_purity_sum += max_pred_intersection
    inverse_purity = inverse_purity_sum / sum(len(c) for c in true)

    f_score = (
        (2 * purity * inverse_purity) / (purity + inverse_purity)
        if (purity + inverse_purity) > 0
        else 0
    )
    return purity, inverse_purity, f_score
