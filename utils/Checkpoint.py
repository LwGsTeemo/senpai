import torch

from utils.toolkit import check_file_exists, load_path
from config import training_config as cfg

from models.unet import UNet3D
from models.unet3plus import UNet_3Plus
from models.unet_revised import UNet_revised
from models.unet_CBAM import UNet_CBAM
from models.unet3plus_CBAM import UNet_3Plus_CBAM

from models.att_grid_unet.unet_grid_att_3D import unet_CT_multi_att_dsv_3D
from models.unet3D_abdomen_cascade import cascade_3DUNet
from models import med3d

from models.UNETR.unetr import UNETR
from models.Transunet.transunet import VisionTransformer, CONFIGS

def load_model(model_name, input_shape, n_classes):
    """ Load model with func `model_link`.
    """
    model = model_link(model_name, input_shape, cfg.dataset.channels, n_classes)
    return model

def model_link(model, input_shape, channel, n_classes):
    """ Get the model using "cfg.model_name" and other arguments from `utils/utils.py`
    """
    if model == "UNET":
        return UNet3D(channel, n_classes)
    elif model == "UNET3+" or model == "UNET3+_DS":
        return UNet_3Plus(channel, n_classes, is_deconv=cfg.deep_supervision)
    elif model == "UNET_CBAM" or model == "UNET_CBAM_DS":
        return UNet_CBAM(channel, n_classes, deep_supervision=cfg.deep_supervision)
    elif model == "UNET3+_CBAM" or model == "UNET3+_DS_CBAM":
        return UNet_3Plus_CBAM(channel, n_classes, is_deconv=cfg.deep_supervision)
    elif model == "UNET_CBAM_pool_DS" or model == "UNET_CBAM_pool" or model == "UNET_pool_DS":
        return UNet_revised(channel, n_classes, deep_supervision=cfg.deep_supervision)
    elif model == "Att_UNet":
        return unet_CT_multi_att_dsv_3D(n_classes=n_classes, in_channels=channel)
    elif model == "cascade_3Dunet":
        return cascade_3DUNet(channel, n_classes)
    elif model == "Med3D":
        net = med3d.resnet34(sample_input_W=input_shape[1], sample_input_H=input_shape[0],
                sample_input_D=input_shape[2], shortcut_type='B', no_cuda=False, num_seg_classes=n_classes)
        net, _ = med3d.load_pretrain(net, './pretrain/resnet_34_23dataset.pth')
        return net
    elif model == "UNETR":
        return UNETR(input_shape, channel, n_classes)
    elif model == "TransUNet":
        return VisionTransformer(CONFIGS['R50-ViT-B_16'], num_classes=n_classes)
    else:
        raise ValueError("Wrong Model Name.")
    
def load_ckpt(model, checkpoint_path, optimizer=None, is_best=False, k_fold=True):
    """ Load pretrain checkpoint (model weights) from the saved file.
    """
    init_epoch, counter_init, best_acc = 0, 0, 0.
    check_file_exists(checkpoint_path)
    file_list = load_path(checkpoint_path)
    if is_best:
        ckpt_list = [file for file in file_list if 'best' in file]
    else:
        ckpt_list = [file for file in file_list if '.pth' in file and 'best' not in file]
        ckpt_list.sort()
    if ckpt_list:
        checkpoint_path = ckpt_list[-1]
        checkpoint = torch.load(checkpoint_path)
        if k_fold:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint
        if cfg.training.data_parallel:
            pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        model.load_state_dict(pretrained_dict)
        if k_fold:
            best_acc = checkpoint['best_acc']
            init_epoch = checkpoint['epoch']
        if 'counter' in checkpoint:
            counter_init = checkpoint['counter']
        if optimizer is not None:   # testing is without an optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("[INFO] Load model weight from {}.".format(checkpoint_path))
    else:
        print("[INFO] Creating a new model.")
    return model, init_epoch, best_acc, optimizer, counter_init

def save_ckpt(epoch, model, optimizer, best_acc, save_path, counter):
    """ Save checkpoint.

    Args:
        epoch : the epoch rn
        counter : used to count the steps for early stoppping criteria
    """
    ckpt = {
        'epoch': epoch + 1,
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'counter': counter
    }
    torch.save(ckpt, save_path)