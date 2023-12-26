import torch
# from sklearn.metrics import f1_score
from sklearn.metrics import f1_score as calcualte_f1_score


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=2):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def f1_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target) 
        a=calcualte_f1_score(target.cpu().numpy(), pred.cpu().numpy(), average='macro')
    return calcualte_f1_score(target.cpu().numpy(), pred.cpu().numpy(), average='macro')
