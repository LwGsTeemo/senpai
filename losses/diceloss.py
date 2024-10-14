import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        
        inputs = inputs[:, 0].contiguous().view(-1)
        targets = targets[:, 0].contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1. - dice

class MulticlassDiceLoss(nn.Module):
    def __init__(self, batch=True):
        super(MulticlassDiceLoss, self).__init__()
        self.batch = batch
    
    def forward(self, input, target, smooth=1):
        input = F.softmax(input, dim=1)
        # 0: background
        target = target[:, 1:]
        input = input[:, 1:]
        
        reduce_axis = torch.arange(2, len(input.shape)).tolist()    
        if self.batch:
            reduce_axis = [0] + reduce_axis
        
        intersection = torch.sum(input * target, dim=reduce_axis)
        dice_loss = 1. - ((2. * intersection + smooth) / (torch.sum(input, dim=reduce_axis) + torch.sum(target, dim=reduce_axis) + smooth))
        # mean of classes per batch
        return dice_loss.mean()