import os
import numpy as np
from tqdm import tqdm
import torch
from torch import from_numpy
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.toolkit import save_cache, load_path
from utils.preprocessing import read_nii_file, clip, normalize, resize_volume, detect_by_mask, crop_volume, clahe_3d, resample, find_uniform_bbx
from utils.Checkpoint import load_model, load_ckpt
from config import training_config as cfg

class ImageDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None, size=(192, 128, 128), n_class=2):
        self.dataset_name = image_path.split('/')[1]
        self.img_path = image_path
        self.label_path = label_path
        self.img_paths = load_path(self.img_path)
        self.label_paths = load_path(self.label_path)
        
        self.size = size
        self.n_class = n_class
        self.transform = transform
        print(self.transform)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_dataset(self, save_img, save_label):
        """ Done preprocessing to data and save them to cache.
        Allow training phase fetch the data from the cache path.
        """
        for j in tqdm(range(len(self.img_paths))):
            img_volume, _, header = read_nii_file(self.img_paths[j])
            label_volume, _, _ = read_nii_file(self.label_paths[j])
            
            origin_space = (header['pixdim'][1], header['pixdim'][2], header['pixdim'][3])
            
            """ Preprocess: resample, localization, normalize, resize """
            img_volume = resample(img_volume, origin_space, cfg.dataset.resample_slice)
            label_volume = resample(label_volume, origin_space, cfg.dataset.resample_slice, label=True)
            
            bbx = detect_by_mask(label_volume, 0)
            real_bbx = find_uniform_bbx(label_volume, bbx, self.size, origin_space, cfg.dataset.resample_slice)
            img_volume, label_volume = crop_volume(img_volume, label_volume, real_bbx)
            
            img_volume = clip(img_volume, -300, 500)
            if cfg.dataset.contrast_enhance:
                img_volume = clahe_3d(img_volume.astype(np.uint8), 0.0)
                img_volume = 255 - img_volume
            img_volume = normalize(img_volume)
            img_volume = np.expand_dims(img_volume, 0)  # (h, w, d) - > (c, h, w, d)
            
            save_cache(img_volume, os.path.join(save_img + self.img_paths[j].split('/')[-1]), True)
            save_cache(label_volume, os.path.join(save_label + self.label_paths[j].split('/')[-1]), True)
        
        self.img_paths = load_path(save_img + self.img_path.split('/')[-1])
        self.label_paths = load_path(save_label + self.label_path.split('/')[-1])
    
    def create_test_data(self, root, method):
        """ Used in evaluate.py. Not being used in k-fold method.
        """
        print("[INFO] Load localization model...")
        local_model_path = "saved_model/Pancreas-CT/full/UNET/"
        detect_model_name = local_model_path.split("/")[-2]
        detect_shape = (128, 128, 64)
        detect_model = load_model(detect_model_name, detect_shape, self.n_class)
        detect_model, _, _, _ = load_ckpt(detect_model, local_model_path, None)
        detect_model = detect_model.to(self.device)
        detect_model.eval()

        # mean_bbx_iou = 0
        with torch.no_grad():
            for j in tqdm(range(len(self.img_paths))):
                img_volume, _, header = read_nii_file(self.img_paths[j])
                label_volume, _, _ = read_nii_file(self.label_paths[j])
                
                """ Preprocess: localization, normalize, resize """
                x_ = clip(img_volume, -300, 500)
                x_ = normalize(x_)
                x_ = resize_volume(x_, detect_shape)
                x_ = np.expand_dims(np.expand_dims(x_, 0), 0)   # (h, w, d) - > (n, c, h, w, d)
                x_ = torch.from_numpy(x_).float().to(self.device)
                
                y_ = detect_model(x_)
                y_ = F.softmax(y_, dim=1)
                y_ = torch.argmax(y_, dim=1)[0].cpu().numpy()
                y_ = resize_volume(y_, label_volume.shape, is_label=True)
                
                """ cal bbx iou """
                # bbx_cal = detect_by_mask(y_, 0)
                # true_bbx = detect_by_mask(label_volume, 0)
                # bbx_iou = iou_3d(bbx_cal, true_bbx)
                # mean_bbx_iou += bbx_iou
                # print("[INFO] {} | bbx iou: {:.4f}".format(self.img_paths[j].split("/")[-1].split(".")[0], bbx_iou))
                
                """ resample """
                origin_space = (header['pixdim'][1], header['pixdim'][2], header['pixdim'][3])
                
                img_volume = resample(img_volume, origin_space, cfg.dataset.resample_slice)
                label_volume = resample(label_volume, origin_space, cfg.dataset.resample_slice, label=True)
                
                """ localization """
                bbx = detect_by_mask(y_, 0)
                real_bbx = find_uniform_bbx(label_volume, bbx, self.size, origin_space, cfg.dataset.resample_slice, transform=True)
                img_volume, label_volume = crop_volume(img_volume, label_volume, real_bbx)
                
                img_volume = clip(img_volume, -300, 500)
                if cfg.dataset.contrast_enhance:
                    img_volume = clahe_3d(img_volume.astype(np.uint8), 0.0)
                    img_volume = 255 - img_volume
                img_volume = normalize(img_volume)
                img_volume = np.expand_dims(img_volume, 0)  # (h, w, d) - > (c, h, w, d)
                
                save_cache(img_volume, os.path.join("cache", root, method.split("/")[0] + self.img_paths[j].split(self.dataset_name)[-1]), True)
                save_cache(label_volume, os.path.join("cache", root, method.split("/")[0] + self.label_paths[j].split(self.dataset_name)[-1]), True)
            
        # mean_bbx_iou /= len(self.img_paths)
        # print("[INFO] mean bbx iou: {:.4f}".format(mean_bbx_iou))
        
        self.img_paths = load_path("cache/" + root + method.split("/")[0] + self.img_path.split(self.dataset_name)[-1])
        self.label_paths = load_path("cache/" + root + method.split("/")[0] + self.label_path.split(self.dataset_name)[-1])
        
    def __getitem__(self, i):
        x = from_numpy(np.load(self.img_paths[i])).float()
        mask = from_numpy(np.load(self.label_paths[i]))
        mask = F.one_hot(mask.long(), num_classes=self.n_class)
        mask = torch.permute(mask, (3, 0, 1, 2)).float()    # (h, w, d, c) - > (c, h, w, d)
        
        data_dicts = {"image": x, "label": mask}
        if self.transform:
            data_dicts = self.transform(data_dicts)
        return data_dicts["image"], data_dicts["label"]
        # return x, mask

    def __len__(self):
        return len(self.img_paths)
