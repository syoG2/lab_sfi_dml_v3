import torch
import torch.nn as nn


def siamese_collate_fn(batch):
    output_dict = {"batch_size": len(batch)}
    for instance in ["instance1", "instance2"]:
        output_dict[instance] = {"verb": [], "frame": [], "ex_idx": []}
        for i in ["input_ids", "token_type_ids", "attention_mask"]:
            if i in batch[0][instance]:
                output_dict[instance][i] = nn.utils.rnn.pad_sequence(
                    [torch.LongTensor(b[instance][i]) for b in batch],
                    batch_first=True,
                )
        output_dict[instance]["target_tidx"] = torch.LongTensor(
            [b[instance]["target_tidx"] for b in batch]
        )
        for b in batch:
            output_dict[instance]["verb"].append(b[instance]["verb"])
            output_dict[instance]["frame"].append(b[instance]["frame"])
            output_dict[instance]["ex_idx"].append(b[instance]["ex_idx"])
    if "label" in batch[0]:
        output_dict["label"] = torch.Tensor([b["label"] for b in batch])
    if "true" in batch[0]:
        output_dict["true"] = torch.Tensor([b["true"] for b in batch])
    return output_dict


def triplet_collate_fn(batch):
    output_dict = {"batch_size": len(batch)}
    for instance in ["anchor", "positive", "negative"]:
        output_dict[instance] = {"verb": [], "frame": [], "ex_idx": []}
        for i in ["input_ids", "token_type_ids", "attention_mask"]:
            if i in batch[0][instance]:
                output_dict[instance][i] = nn.utils.rnn.pad_sequence(
                    [torch.LongTensor(b[instance][i]) for b in batch],
                    batch_first=True,
                )
        output_dict[instance]["target_tidx"] = torch.LongTensor(
            [b[instance]["target_tidx"] for b in batch]
        )
        for b in batch:
            output_dict[instance]["verb"].append(b[instance]["verb"])
            output_dict[instance]["frame"].append(b[instance]["frame"])
            output_dict[instance]["ex_idx"].append(b[instance]["ex_idx"])
    return output_dict


def classification_collate_fn(batch):
    output_dict = {
        "verb": [],
        "frame": [],
        "ex_idx": [],
        "batch_size": len(batch),
    }
    for i in ["input_ids", "token_type_ids", "attention_mask"]:
        if i in batch[0]:
            output_dict[i] = nn.utils.rnn.pad_sequence(
                [torch.LongTensor(b[i]) for b in batch], batch_first=True
            )
    output_dict["target_tidx"] = torch.LongTensor([b["target_tidx"] for b in batch])

    for b in batch:
        output_dict["verb"].append(b["verb"])
        output_dict["frame"].append(b["frame"])
        output_dict["ex_idx"].append(b["ex_idx"])
    if "label" in batch[0]:
        output_dict["label"] = torch.LongTensor([b["label"] for b in batch])
    return output_dict


def base_collate_fn(batch):
    output_dict = {
        "verb": [],
        "frame": [],
        "ex_idx": [],
        "batch_size": len(batch),
    }
    for i in ["input_ids", "token_type_ids", "attention_mask"]:
        if i in batch[0]:
            output_dict[i] = nn.utils.rnn.pad_sequence(
                [torch.LongTensor(b[i]) for b in batch], batch_first=True
            )
    output_dict["target_tidx"] = torch.LongTensor([b["target_tidx"] for b in batch])

    for b in batch:
        output_dict["verb"].append(b["verb"])
        output_dict["frame"].append(b["frame"])
        output_dict["ex_idx"].append(b["ex_idx"])
    return output_dict
