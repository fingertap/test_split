from sklearn.metrics import accuracy_score, roc_auc_score


def accuracy(pred, label):
    # pred: [N, K], label: [N]
    return accuracy_score(label, pred.argmax(dim=-1))


def roc_auc(pred, label):
    # pred: [N], label: [N]
    return roc_auc_score(label, pred)
