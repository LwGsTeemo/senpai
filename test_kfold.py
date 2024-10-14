import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from utils.Checkpoint import load_model, load_ckpt
from dataio.Dataset_v2 import ImageDataset
from utils.metrics import DiceScore
from utils.preprocessing import read_nii_file, clip, normalize, detect_by_mask, crop_volume, clahe_3d, resample, find_uniform_bbx, resize_image_with_crop_or_pad, resample_back
from utils.postprocessing import filter_noise, apply_gaussian_smoothing
from utils.toolkit  import save_result, check_file_exists
from config import training_config as cfg
from models.CRF.crf import CRF

def predict(root, input_shape, NUM_CLASSES, method, model, fold, test_img_path, test_label_path, device="cuda:0", **params):    
    print("[INFO] Load localization model...")
    local_model_path = "saved_model/Pancreas-CT_5fold/full/UNET/" + f'K{fold}/'
    detect_model_name = local_model_path.split("/")[-3]
    detect_shape = (192, 192, 128)
    detect_model = load_model(detect_model_name, detect_shape, NUM_CLASSES)
    detect_model, _, _, _, _ = load_ckpt(detect_model, local_model_path, optimizer=None, is_best=True)
    detect_model = detect_model.to(device)
    
    print("[INFO] Load test data...")
    test_data = ImageDataset(test_img_path, test_label_path, **params)
    
    post_model = CRF(num_iter=10, num_classes=NUM_CLASSES).to(device)
    dice = DiceScore(activation=False)
    
    save_path = "result/" + cfg.dataset.name + method + cfg.model_name + "/" + f'K{fold}/'
    check_file_exists(save_path)
    
    print("--------------\n[INFO] Start Predict...")
    detect_model.eval()
    model.eval()
    test_acc, test_fixed_acc = 0, 0
    with torch.no_grad():
        for X, Y in tqdm(zip(test_data.img_paths, test_data.label_paths), total=len(test_data.img_paths)):
            x, _, header = read_nii_file(X)
            y, y_affine, _ = read_nii_file(Y)
            
            """ resample """
            origin_space = (header['pixdim'][1], header['pixdim'][2], header['pixdim'][3])
            resample_x = resample(x, origin_space, cfg.dataset.resample_slice)
            resample_y = resample(y, origin_space, cfg.dataset.resample_slice, label=True)
            
            """ preprocessing: clip, contrast enhance, normalize """
            x = clip(resample_x, -300, 500)
            # if cfg.dataset.contrast_enhance:
            x = clahe_3d(x.astype(np.uint8), 0.0)
            x = 255 - x
            x = normalize(x)
            
            """ resize: crop/pad """
            x_, slicer = resize_image_with_crop_or_pad(x, detect_shape)
            x_ = np.expand_dims(np.expand_dims(x_, 0), 0)
            
            """ detection/localization model """
            x_ = torch.from_numpy(x_).float().to(device)
            y_ = detect_model(x_)
            y_ = F.softmax(y_, dim=1)
            y_ = torch.argmax(y_, dim=1)[0].cpu().numpy().astype('float32')
            
            """ stick back to the resampled shape """        
            y_detect = np.zeros(resample_y.shape)
            y_detect[slicer[0], slicer[1], slicer[2]] = y_
            assert y_detect.shape == resample_y.shape, "excepted shape {}, got {}.".format(resample_y.shape, y_detect.shape)
            
            """ to separate clahe for detect and seg model"""
            x = clip(resample_x, -300, 500)
            if cfg.dataset.contrast_enhance:
                x = clahe_3d(x.astype(np.uint8), 0.0)
                x = 255 - x
            x = normalize(x)
            
            """ localization and crop """
            bbx = detect_by_mask(y_detect, 0)
            real_bbx = find_uniform_bbx(x, bbx, input_shape, origin_space, cfg.dataset.resample_slice, transform=False)            
            crop_x, _ = crop_volume(x, resample_y, real_bbx)

            """ predict """ 
            x = np.expand_dims(np.expand_dims(crop_x, 0), 0) # (h, w, d) - > (n, c, h, w, d)
            x = torch.from_numpy(x).float().to(device)
            if cfg.deep_supervision:
                pred = model(x)[0]
            else:
                pred = model(x)
            
            """ post-processing: crfasrnn & contour smoothing """
            if cfg.predict.crf:
                crop_x = np.expand_dims(np.expand_dims(crop_x, 0), 0)
                crop_x = torch.from_numpy(crop_x).float().to(device)
                pred = post_model(pred, crop_x)
            else:
                pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)[0].cpu().numpy().astype('float32')
            
            if cfg.predict.smoothing:
                pred = apply_gaussian_smoothing(pred, sigma=2.0, iterations=1)
            
            """ strick to resampled size """
            y_pred = np.zeros(resample_y.shape)
            y_pred[real_bbx[0]:real_bbx[1], real_bbx[2]:real_bbx[3], real_bbx[4]:real_bbx[5]] = pred
            
            """ resample back to the origin spacing """
            if cfg.predict.resample_back:
                # y_pred = resample(y_pred, cfg.dataset.resample_slice, origin_space, label=True)
                y_pred = resample_back(y_pred, y.shape, is_label=True)
                assert y_pred.shape == y.shape, "excepted shape {}, got {}.".format(y.shape, y_pred.shape)
            
            """ post-processing: filter FP """
            y_pred_fixed = np.zeros(y_pred.shape)
            y_pred_fixed = y_pred.copy()
            if cfg.predict.filter_noise:
                y_pred_fixed = filter_noise(y_pred_fixed)
            
            """ cal. for accuracy """
            y_pred = torch.from_numpy(y_pred).float().to(device)
            y_pred_fixed = torch.from_numpy(y_pred_fixed).float().to(device)
            if cfg.predict.resample_back:
                y = torch.from_numpy(y).float().to(device)
                acc = dice(y_pred, y).item()
                acc_fixed = dice(y_pred_fixed, y).item()
            else:
                resample_y = torch.from_numpy(resample_y).float().to(device)
                acc = dice(y_pred, resample_y).item()
                acc_fixed = dice(y_pred_fixed, resample_y).item()
            test_acc += acc
            test_fixed_acc += acc_fixed
            # print("{} | dice: {:.2f}, fixed: {:.2f}".format(X.split('/')[-1].split('.')[0], acc, acc_fixed))
            
            """ save as nii """
            y_pred = y_pred.cpu().numpy().astype('float32')
            y_pred_fixed = y_pred_fixed.cpu().numpy().astype('float32')
            save_result(y_pred, os.path.join(save_path, "predict_" + X.split('/')[-1]), affine=y_affine)
            save_result(y_pred_fixed, os.path.join(save_path, "result_" + X.split('/')[-1]), affine=y_affine)

    test_acc /= len(test_data.img_paths)
    test_fixed_acc /= len(test_data.img_paths)
    return test_acc, test_fixed_acc
