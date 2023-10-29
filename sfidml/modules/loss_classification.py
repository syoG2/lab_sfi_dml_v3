import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.training:
            with torch.no_grad():
                B_avg = torch.where(
                    one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits)
                )
                B_avg = torch.sum(B_avg) / input.size(0)
                theta_med = torch.median(theta[one_hot == 1])
                self.s = torch.log(B_avg) / torch.cos(
                    torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med)
                )
        output = self.s * logits
        return output


class ArcFace(nn.Module):
    def __init__(self, num_features, num_classes, s=64, m=0.50):
        super(ArcFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)

        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        return output
