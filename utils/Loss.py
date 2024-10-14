import torch.nn as nn
import torch.nn.functional as F

""" combine two or above types of loss functions; the code of each loss function is in losses/ 
"""

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        from losses.diceloss import DiceLoss
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        diceLoss = self.dice_loss(pred, target)
        pred = F.sigmoid(pred)
        BCE = F.binary_cross_entropy(pred, target, reduction='mean')
        return BCE + diceLoss

class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        from losses.diceloss import MulticlassDiceLoss
        self.dice_loss = MulticlassDiceLoss()

    def forward(self, pred, target):
        CE = F.cross_entropy(pred, target, reduction='mean')
        return CE + self.dice_loss(pred, target)

class hybrid_loss(nn.Module):
    def __init__(self):
        super().__init__()
        from losses.cross_entropy import FocalLoss
        self.focal = FocalLoss()
        from losses.msssim import MSSSIM3D
        self.ms_ssim = MSSSIM3D()
        from losses.iou_loss import IoULoss
        self.jacard = IoULoss()
    
    def forward(self, pred, target):
        f_loss = self.focal(pred, target)
        ssim_loss = self.ms_ssim(pred, target)
        iou_loss = self.jacard(pred, target)
        return f_loss + ssim_loss + iou_loss

class multiLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(multiLoss, self).__init__()
        from losses.diceloss import MulticlassDiceLoss
        self.dice_loss = MulticlassDiceLoss()
        from losses.cross_entropy import FocalLoss
        self.focal_loss = FocalLoss()

    def forward(self, pred, target):
        return self.focal_loss(pred, target) + self.dice_loss(pred, target)