import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score
from focal_loss.focal_loss import FocalLoss
import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = torch.nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def nll_loss(output, target):
    return F.nll_loss(output, target)

def f1_score_loss(input, target):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f1 = MulticlassF1Score(num_classes=18, average=None).to(device)
    return f1(input, target).requires_grad_(True).mean()

def focal_loss(input, target):
    # criterion = FocalLoss(gamma=0.7)
    # batch_size = 64
    # n_class = 18
    # m = F.softmax(dim=-1)
    # logits = torch.randn(batch_size, n_class)
    # target = torch.randn(0, n_class, size=(batch_size,))
    # return criterion(m(logits), target)
    focal = FocalLoss()
    return focal(input, target)

def labelSmoothingLoss(input, target):
    model = LabelSmoothingLoss()
    return model(input, target)

def corss_entropy(input, target):
    return F.cross_entropy(input, target)