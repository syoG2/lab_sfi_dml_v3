import argparse
from pathlib import Path
from socket import gethostname
from time import time

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from sfidml.f_induc.collate_fn import (
    classification_collate_fn,
    siamese_collate_fn,
    triplet_collate_fn,
)
from sfidml.f_induc.dataset import (
    ClassificationDataset,
    SiameseDataset,
    TripletDataset,
)
from sfidml.f_induc.embedding import BaseEmbedding
from sfidml.f_induc.model import (
    BaseNet,
    ClassificationNet,
    SiameseNet,
    TripletNet,
)
from sfidml.f_induc.ranking import run_ranking
from sfidml.utils.data_utils import read_jsonl, write_json, write_jsonl
from sfidml.utils.model_utils import fix_seed


def step(dl, model, optimizer=None):
    start = time()
    total_loss, count = 0, 0
    for batch in tqdm(dl):
        if model.training:
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        else:
            loss = model.compute_loss(batch)
        total_loss += loss.item() * batch["batch_size"]
        count += batch["batch_size"]
    total_loss /= count
    end = time()

    metrics = {"loss": total_loss, "time": end - start}
    if model.training:
        return metrics, model, optimizer
    else:
        return metrics


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.DataFrame(read_jsonl(args.input_train_file))
    # df_dev = pd.DataFrame(read_jsonl(args.input_dev_file))

    if "vanilla" in args.model_name:
        model = BaseNet(
            args.pretrained_model_name,
            args.normalization,
            args.device,
        )
    else:
        if "siamese_distance" in args.model_name:
            ds_train = SiameseDataset(
                df_train,
                args.pretrained_model_name,
                args.vec_type,
            )
            model = SiameseNet(
                args.pretrained_model_name,
                args.normalization,
                args.margin,
                args.device,
            )
            collate_fn = siamese_collate_fn
        elif "triplet_distance" in args.model_name:
            ds_train = TripletDataset(
                df_train,
                args.pretrained_model_name,
                args.vec_type,
            )
            model = TripletNet(
                args.pretrained_model_name,
                args.normalization,
                args.margin,
                args.device,
            )
            collate_fn = triplet_collate_fn
        elif "classification" in args.model_name:
            ds_train = ClassificationDataset(
                df_train, args.pretrained_model_name, args.vec_type
            )
            model = ClassificationNet(
                args.pretrained_model_name,
                args.model_name,
                args.normalization,
                args.margin,
                args.device,
                len(ds_train.frame2label),
            )
            collate_fn = classification_collate_fn
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    log_list = []
    fix_seed(0)
    if args.model_name != "vanilla":
        for epoch in range(1, args.n_epochs + 1):
            if "distance" in args.model_name:
                ds_train.make_dataset()

            dl_train = DataLoader(
                ds_train,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )

            model.to(args.device).train()
            metrics_train, model, optimizer = step(dl_train, model, optimizer)
            # embedding = BaseEmbedding(
            #     model.pretrained_model,
            #     args.pretrained_model_name,
            #     args.vec_type,
            #     args.batch_size,
            # )
            # df_vec, vec_array = embedding.get_embedding(df_dev)
            # ranking_scores_dev = run_ranking(df_vec, vec_array)

            log_dict = {"epoch": epoch, "time": time()}
            log_dict.update({f"train-{k}": v for k, v in metrics_train.items()})
            # log_dict.update(
            #     {f"dev-{k}": v for k, v in ranking_scores_dev.items()}
            # )
            log_list.append(log_dict)
            torch.save(
                optimizer.state_dict(), args.output_dir / "optimizer.pth"
            )
    else:
        # embedding = BaseEmbedding(
        #         model.pretrained_model,
        #         args.pretrained_model_name,
        #         args.vec_type,
        #         args.batch_size,
        #     )
        # df_vec, vec_array = embedding.get_embedding(df_dev)
        # ranking_scores_dev = run_ranking(df_vec, vec_array)
        # log_dict = {"dev-" + k: v for k, v in ranking_scores_dev.items()}
        log_dict = {"time": time()}
        log_list.append(log_dict)

    write_jsonl(log_list, args.output_dir / "metrics.jsonl")
    write_json(vars(args), args.output_dir / "params.json")
    if "classification" in args.model_name:
        write_json(ds_train.frame2label, args.output_dir / "frame2label.json")

    torch.save(
        model.pretrained_model.to("cpu").state_dict(),
        args.output_dir / "pretrained_model_last.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_file", type=Path, required=True)
    parser.add_argument("--input_dev_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument(
        "--pretrained_model_name", type=str, default="bert-base-uncased"
    )
    parser.add_argument("--model_name", type=str, default="siamese_distance")
    parser.add_argument("--vec_type", type=str, default="word")
    parser.add_argument("--run_number", type=str, default="00")

    parser.add_argument("--normalization", type=str, default="true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=5)

    parser.add_argument("--margin", type=float, default=1)
    args = parser.parse_args()
    if args.model_name in ["vanilla"]:
        for k in ["margin", "learning_rate", "n_epochs"]:
            args.__delattr__(k)

    if args.model_name in ["softmax_classification", "adacos_classification"]:
        args.margin = None

    args.server = gethostname()
    print(args)
    main(args)
