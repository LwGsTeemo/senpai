import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceScore(nn.Module):
    def __init__(self, activation=True):
        super(DiceScore, self).__init__()
        self.activation = activation

    def forward(self, inputs, targets, smooth=1):
        if self.activation:
            inputs = F.sigmoid(inputs)
        
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return dice

class MulticlassDiceScore(nn.Module):
    def __init__(self, activation=True, batch=True):
        super(MulticlassDiceScore, self).__init__()
        self.batch = batch
        self.activation = activation
    
    def forward(self, input, target, smooth=1):
        if self.activation:
            input = F.softmax(input, dim=1)
        # 0: background
        target = target[:, 1:]
        input = input[:, 1:]
        
        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis
        
        intersection = torch.sum(input * target, dim=reduce_axis)
        dice = ((2. * intersection + smooth) / (torch.sum(input, dim=reduce_axis) + torch.sum(target, dim=reduce_axis) + smooth))
        # mean of classes per batch
        return dice.mean()

class PrecisionRecall(nn.Module):
    def __init__(self, activation=True):
        super(PrecisionRecall, self).__init__()
        self.activation = activation

    def forward(self, pred, target, threshold=0.5, smooth=1e-10):
        if self.activation:
            pred = F.sigmoid(pred)
            
        pred = (pred >= threshold).float()
        tp = (pred * target).sum(dim=(0, 1, 2))
        fp = (pred * (1 - target)).sum(dim=(0, 1, 2))
        fn = ((1 - pred) * target).sum(dim=(0, 1, 2))
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        return precision, recall