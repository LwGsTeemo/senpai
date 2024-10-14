from easydict import EasyDict

def pancreas_dataset(cfg):
    cfg.dataset = EasyDict()
    cfg.dataset.name = "Pancreas-CT/"
    cfg.dataset.root = "dataset/" + cfg.dataset.name
    cfg.dataset.train_img_path = "train/image/"
    cfg.dataset.train_label_path = "train/label/"
    cfg.dataset.val_img_path = "val/image/"
    cfg.dataset.val_label_path = "val/label/"

    cfg.dataset.test_img_path = "test/image/"
    cfg.dataset.test_label_path = "test/label/"
    
    cfg.dataset.channels = 1
    cfg.dataset.num_class = 2
    return cfg

def K5_pancreas_dataset(cfg):
    cfg.dataset = EasyDict()
    # cfg.dataset.name = "Pancreas-CT_5fold/"
    cfg.dataset.name = "Task07_Pancreas/"
    cfg.dataset.root = "dataset/" + cfg.dataset.name
    cfg.dataset.train_img_path = "train/image/"
    cfg.dataset.train_label_path = "train/label/"

    cfg.dataset.test_img_path = "test/image/"
    cfg.dataset.test_label_path = "test/label/"
    
    cfg.dataset.channels = 1
    cfg.dataset.num_class = 2
    return cfg

def unet_3d_config():
    cfg = EasyDict()
    cfg.learning_rate = 3e-4
    cfg.weight_decay = 1e-5
    cfg.momentum = 0.0

    cfg.model_name = "UNET"
    cfg.optimizer = "Adam"
    cfg.loss = "DiceCE"
    cfg.deep_supervision = False
    return cfg

def unet3plus_3d_config():
    cfg = EasyDict()
    cfg.learning_rate = 3e-4
    cfg.weight_decay = 1e-5
    cfg.momentum = 0.0

    cfg.model_name = "UNET3+"
    cfg.optimizer = "Adam"
    cfg.loss = "SSIM"
    cfg.deep_supervision = False
    return cfg

def unet3plus_DS_3d_config():
    cfg = EasyDict()
    cfg.learning_rate = 3e-4
    cfg.weight_decay = 1e-5
    cfg.momentum = 0.0

    cfg.model_name = "UNET3+_DS"
    cfg.optimizer = "Adam"
    cfg.loss = "SSIM"
    cfg.deep_supervision = True
    return cfg

def transunet_3d_config():
    cfg = EasyDict()
    cfg.learning_rate = 5e-4
    cfg.weight_decay = 1e-5
    cfg.momentum = 0.9

    cfg.model_name = "TransUNet"
    cfg.optimizer = "Adam"
    cfg.loss = "DiceCE"
    cfg.deep_supervision = False
    return cfg

def unetr_config():
    cfg = EasyDict()
    cfg.learning_rate = 3e-4
    cfg.weight_decay = 1e-5
    cfg.momentum = 0.0

    cfg.model_name = "UNETR"
    cfg.optimizer = "AdamW"
    cfg.loss = "DiceCE"
    cfg.deep_supervision = False
    return cfg

def stunet_config():
    cfg = EasyDict()
    cfg.learning_rate = 1e-3
    cfg.weight_decay = 3e-5
    cfg.momentum = 0.99

    cfg.model_name = "STUNet"
    cfg.optimizer = "SGD"
    cfg.loss = "DiceCE"
    cfg.deep_supervision = True
    return cfg

def unet_cbam_config():
    cfg = EasyDict()
    cfg.learning_rate = 3e-4
    cfg.weight_decay = 1e-5
    cfg.momentum = 0.0

    cfg.model_name = "UNET_CBAM"
    cfg.optimizer = "Adam"
    cfg.loss = "DiceCE"
    cfg.deep_supervision = False
    return cfg

def att_grid_config():
    cfg = EasyDict()
    cfg.learning_rate = 1e-4
    cfg.weight_decay = 1e-5
    cfg.momentum = 0.0

    cfg.model_name = "Att_UNet"
    cfg.optimizer = "Adam"
    cfg.loss = "Dice"
    cfg.deep_supervision = False
    return cfg

def cascade_unet_3d_config():
    cfg = EasyDict()
    cfg.learning_rate = 1e-4
    cfg.weight_decay = 1e-5
    cfg.momentum = 0.0

    cfg.model_name = "cascade_3Dunet"
    cfg.optimizer = "Adam"
    cfg.loss = "weightedCE"
    cfg.deep_supervision = False
    return cfg

def med3d_config():
    cfg = EasyDict()
    cfg.learning_rate = 0.1
    cfg.weight_decay = 0.001
    cfg.momentum = 0.9

    cfg.model_name = "Med3D"
    cfg.optimizer = "SGD"
    cfg.loss = "CE"
    cfg.deep_supervision = False
    return cfg

def unet_cbam_pp_config():
    cfg = EasyDict()
    cfg.learning_rate = 3e-4
    cfg.weight_decay = 1e-5
    cfg.momentum = 0.0

    cfg.model_name = "UNET_CBAM_pool_DS"
    cfg.optimizer = "Adam"
    cfg.loss = "DiceCE" # DiceFocal, SSIM
    cfg.deep_supervision = True
    return cfg

CONFIGS = {
    'UNET': unet_3d_config(),
    'UNET3+': unet3plus_3d_config(),
    'UNET3+_DS': unet3plus_DS_3d_config(),
    'UNET_CBAM': unet_cbam_config(),
    'UNET_CBAM_pool_DS': unet_cbam_pp_config(),
    
    'UNETR': unetr_config(),
    'TransUNet': transunet_3d_config(),
    
    'STUNet': stunet_config(),
    'Att_UNet': att_grid_config(),
    'cascade_3Dunet': cascade_unet_3d_config(),
    'Med3D': med3d_config(),
}