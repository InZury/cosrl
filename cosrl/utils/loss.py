import torch
import torch.nn as nn


class DiceLoss:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def __call__(self, predict, target):
        probs = torch.sigmoid(predict).flatten(1)
        targets = target.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.epsilon) / (denominator + self.epsilon)

        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, epsilon=1e-6, from_logits=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.from_logits = from_logits

    def forward(self, predict, target):
        probs = torch.sigmoid(predict) if self.from_logits else predict
        probs = probs.flatten(1)
        target = target.flatten(1)

        tp = (probs * target).sum(dim=1)
        fp = (probs * (1 - target)).sum(dim=1)
        fn = ((1 - probs) * target).sum(dim=1)

        tversky = (tp + self.epsilon) / (tp + self.alpha * fp + self.beta * fn + self.epsilon)

        return 1.0 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, epsilon=1e-6, from_logits=True):
        super().__init__()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta, epsilon=epsilon, from_logits=from_logits)
        self.gamma = gamma

    def forward(self, predict, target):
     tversky_loss =self.tversky_loss(predict, target)

     return tversky_loss.pow(self.gamma)