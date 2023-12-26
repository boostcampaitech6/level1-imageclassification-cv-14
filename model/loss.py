import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        probs = F.softmax(outputs, dim=1)
        class_mask = F.one_hot(targets, num_classes=probs.size(1))

        probs = (probs * class_mask).sum(dim=1)
        focal_loss = -self.alpha * (1 - probs) ** self.gamma * torch.log(probs)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()

        self.epsilon = epsilon

    def forward(self, outputs, targets):
        assert outputs.ndim == 2
        assert targets.ndim == 1

        num_classes = outputs.shape[1]

        preds = F.softmax(outputs, dim=1)
        true = F.one_hot(targets, num_classes).to(torch.float32)

        tp = (true * preds).sum(dim=0).to(torch.float32)
        fp = ((1 - true) * preds).sum(dim=0).to(torch.float32)
        fn = (true * (1 - preds)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        
        return 1 - f1.mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, outputs, targets):
        num_classes = outputs.shape[1]

        preds = outputs.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(preds)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * preds, dim=self.dim))


def nll_loss(outputs, targets):
    return F.nll_loss(outputs, targets)

def cross_entropy_loss(outputs, targets):
    return F.cross_entropy(outputs, targets)

def focal_loss(outputs, targets):
    return FocalLoss()(outputs, targets)

def f1_loss(outputs, targets):
    return F1Loss()(outputs, targets)

def label_smoothing_loss(outputs, targets):
    return LabelSmoothingLoss()(outputs, targets)

def f1_focal_loss(outputs, targets):
    return  0.5*F1Loss()(outputs, targets) + 0.5*FocalLoss()(outputs, targets)

def f1_label_loss(outputs, targets):
    return  0.5*F1Loss()(outputs, targets) + 0.5*LabelSmoothingLoss()(outputs, targets)

def f1_cross_loss(outputs, targets):
    return  0.5*F1Loss()(outputs, targets) + 0.5*F.cross_entropy(outputs, targets)

def focal_cross_loss(outputs, targets):
    return  0.5*FocalLoss()(outputs, targets) + 0.5*F.cross_entropy(outputs, targets)

def focal_label_loss(outputs, targets):
    return  0.5*FocalLoss()(outputs, targets) + 0.5*LabelSmoothingLoss()(outputs, targets)

def label_cross_loss(outputs, targets):
    return 0.5*LabelSmoothingLoss()(outputs, targets) + 0.5*F.cross_entropy(outputs, targets)