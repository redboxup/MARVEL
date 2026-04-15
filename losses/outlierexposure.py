import torch
from torch import nn

class OELoss(nn.Module):
    def forward(self, logits: torch.Tensor):
        return -(logits.mean(1) - torch.logsumexp(logits, dim=1)).mean()
