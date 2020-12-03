
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report


class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class KNNClassification(nn.Module):

    def __init__(self, X_train, Y_true, K=10):
        super().__init__()

        self.K = K

        self.KNN = KNeighborsClassifier(n_neighbors=self.K, weights='distance')
        self.KNN.fit(X_train, Y_true)

    def forward(self, X_test, y_true):

        y_pred = self.KNN.predict(X_test)

        acc = accuracy_score(y_true, y_pred)

        return acc


class calssification_report(nn.Module):

    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names
    def forward(self, predict_labels, true_labels):

        report = classification_report(true_labels, predict_labels, target_names=self.target_names, output_dict=True)

        return report


class TruncatedNSM(nn.Module):
    def __init__(self, dim, nb_class, nb_train_samples, temperature=0.05, q=0.7, k=0.5):
        super().__init__()

        self.cls_weights = Parameter(torch.Tensor(nb_class, dim))
        stdv = 1. / math.sqrt(self.cls_weights.size(1))
        self.cls_weights.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.q = q
        self.k = k
        self.weight = Parameter(data=torch.ones(nb_train_samples, 1), requires_grad=False)

    def forward(self, embeddings, targets, indexes):

        norm_weight = nn.functional.normalize(self.cls_weights, dim=1)
        logits = nn.functional.linear(embeddings, norm_weight)
        logits /= self.temperature

        p = F.softmax(logits, dim=1)

        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss


    def update_weight(self, embeddings, targets, indexes):
        
        norm_weight = nn.functional.normalize(self.cls_weights, dim=1)
        logits = nn.functional.linear(embeddings, norm_weight)
        logits /= self.temperature

        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)








