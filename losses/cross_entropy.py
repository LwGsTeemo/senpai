import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        CE = F.cross_entropy(pred, target, reduction='mean')
        return CE

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        BCE = F.binary_cross_entropy(pred, target, reduction='mean')
        return BCE

class weightedCELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred, target, ):
        CE = F.cross_entropy(pred, target, weight=self.weight, reduction='mean')
        return CE

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        
        CE = F.cross_entropy(inputs, targets, reduction='mean')
        CE_EXP = torch.exp(-CE)
        focal_loss = alpha * (1 - CE_EXP)**gamma * CE
        return focal_loss