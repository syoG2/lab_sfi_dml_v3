import torch
import torch.nn as nn
import torch.nn.functional as F
from fadml.modules.loss_classification import AdaCos, ArcFace
from fadml.modules.loss_distance import ContrastiveLoss, TripletLoss
from transformers import AutoConfig, AutoModel


class BaseNet(nn.Module):
    def __init__(self, pretrained_model_name, normalization, device, layer=-1):
        super(BaseNet, self).__init__()
        config = AutoConfig.from_pretrained(
            pretrained_model_name, output_hidden_states=True
        )
        self.pretrained_model = AutoModel.from_pretrained(
            pretrained_model_name, config=config
        ).to(device)

        self.normalization = normalization
        self.layer = layer
        self.device = device

    def forward(self, inputs):
        if "token_type_ids" in inputs:
            hidden_states = self.pretrained_model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                token_type_ids=inputs["token_type_ids"].to(self.device),
            )["hidden_states"][self.layer]
        else:
            hidden_states = self.pretrained_model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
            )["hidden_states"][self.layer]
        embeddings = hidden_states[
            torch.LongTensor(range(len(inputs["target_tidx"]))),
            inputs["target_tidx"],
        ]
        if self.normalization == "true":
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


class SiameseNet(BaseNet):
    def __init__(self, pretrained_model_name, normalization, margin, device):
        super(SiameseNet, self).__init__(
            pretrained_model_name, normalization, device
        )
        self.margin = margin
        self.criterion = ContrastiveLoss(margin)

    def compute_loss(self, inputs):
        embeddings1 = self.forward(inputs["instance1"])
        embeddings2 = self.forward(inputs["instance2"])
        loss = self.criterion(
            embeddings1, embeddings2, inputs["label"].to(self.device)
        )
        return loss


class TripletNet(BaseNet):
    def __init__(self, pretrained_model_name, normalization, margin, device):
        super(TripletNet, self).__init__(
            pretrained_model_name, normalization, device
        )
        self.margin = margin
        self.criterion = TripletLoss(self.margin)

    def compute_loss(self, inputs):
        anchor_embs = self.forward(inputs["anchor"])
        positive_embs = self.forward(inputs["positive"])
        negative_embs = self.forward(inputs["negative"])
        loss = self.criterion(anchor_embs, positive_embs, negative_embs).mean()
        return loss


class ClassificationNet(BaseNet):
    def __init__(
        self,
        pretrained_model_name,
        model_name,
        normalization,
        margin,
        device,
        label_size,
    ):
        super(ClassificationNet, self).__init__(
            pretrained_model_name, normalization, device
        )
        self.model_name = model_name
        self.margin = margin
        self.label_size = label_size
        self.hidden_size = self.pretrained_model.config.hidden_size

        self.criterion = nn.CrossEntropyLoss()

        if "softmax" in self.model_name:
            self.metric_fc = nn.Linear(self.hidden_size, self.label_size)
        elif "adacos" in self.model_name:
            self.metric_fc = AdaCos(self.hidden_size, self.label_size)
        elif "arcface" in self.model_name:
            self.metric_fc = ArcFace(
                self.hidden_size, self.label_size, m=margin
            )

    def compute_loss(self, inputs):
        embeddings = self.forward(inputs)
        if "softmax" in self.model_name:
            outputs = self.metric_fc(embeddings)
        else:
            outputs = self.metric_fc(
                embeddings, inputs["label"].to(self.device)
            )
        loss = self.criterion(outputs, inputs["label"].to(self.device))
        return loss
