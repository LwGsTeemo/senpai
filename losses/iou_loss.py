import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, weight=None, batch=True):
        super(IoULoss, self).__init__()
        self.batch = batch

    def forward(self, inputs, target, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        # traget 0: background
        target = target[:, 1:]
        inputs = inputs[:, 1:]
        
        reduce_axis = torch.arange(2, len(inputs.shape)).tolist()    
        if self.batch:
            reduce_axis = [0] + reduce_axis
        
        intersection = torch.sum(inputs * target, dim=reduce_axis)
        total = (inputs + target).sum(dim=reduce_axis)
        union = total - intersection
        iou_loss = 1 - (intersection + smooth) / (union + smooth)
        # mean of classes per batch
        return iou_loss.mean()