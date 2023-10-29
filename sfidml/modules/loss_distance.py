import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-12

    def forward(self, output1, output2, label):
        label = label.float()
        distances = (output2 - output1).pow(2).sum(1)
        distance_negative = F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        loss = 0.5 * (label * distances + (1 + -1 * label) * distance_negative).mean()
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        loss = F.relu(distance_positive - distance_negative + self.margin).mean()
        return loss
