import os
import torch
import numpy as np
import json
import torch.nn.functional as F

from utils.Checkpoint import make_or_restore_model
from dataio.Dataset import ImageDataset
from utils.metrics import DiceScore
from utils.preprocessing import read_nii_file, clip, normalize, resize_volume, detect_by_mask, crop_volume, clahe_3d
from utils.postprocessing import filter_noise
from utils.toolkit  import save_result, check_file_exists
from config import training_config as cfg
from models.CRF.crf import CRF

def main():
    root = cfg.dataset.root
    test_img_path = cfg.dataset.test_img_path
    test_label_path = cfg.dataset.test_label_path

    input_shape = cfg.input_shape
    NUM_CLASSES = cfg.num_class
    method = cfg.method

    print("[INFO] Load test data...")
    params = {
        'size': input_shape,
        'n_class': NUM_CLASSES
    }
    test_data = ImageDataset(root + test_img_path, root + test_label_path, **params)
    
    print("[INFO] Load localization model...")
    local_model_path = "saved_model/" + cfg.dataset.name + "full/UNET/"
    detect_model_name = local_model_path.split("/")[-2]
    detect_shape = (128, 128, 64)
    detect_model, _ = make_or_restore_model(detect_model_name, detect_shape, NUM_CLASSES, local_model_path)
    detect_model = detect_model.to(device)
    
    print("[INFO] Load model...")
    checkpoint_path = cfg.checkpoint_path
    model, _ = make_or_restore_model(cfg.model_name, input_shape, NUM_CLASSES, checkpoint_path)
    model = model.to(device)
    post_model = CRF(num_iter=10, num_classes=NUM_CLASSES).to(device)
    
    dice = DiceScore(activation=False)
    
    save_path = "result/" + cfg.dataset.name + method + cfg.model_name + "/"
    check_file_exists(save_path)
    bbx_to_file = {}
    
    print("--------------\n[INFO] Start Predict...")
    detect_model.eval()
    model.eval()
    test_acc, test_fixed_acc = 0, 0
    with torch.no_grad():
        for X, Y in zip(test_data.img_paths, test_data.label_paths):
            x, _, _ = read_nii_file(X)
            y, y_affine, _ = read_nii_file(Y)
        
            """ localization """
            x_ = clip(x, -300, 500)
            x_ = normalize(x_)
            x_ = resize_volume(x_, detect_shape)
            # (h, w, d) - > (n, c, h, w, d)
            x_ = np.expand_dims(np.expand_dims(x_, 0), 0)
            x_ = torch.from_numpy(x_).float().to(device)

            y_ = detect_model(x_)
            y_ = F.softmax(y_, dim=1)
            y_ = torch.argmax(y_, dim=1)[0].cpu().numpy()
            y_ = resize_volume(y_, y.shape, is_label=True)
            assert y_.shape == y.shape, "excepted shape {}, got {}.".format(y.shape, y_.shape)
            
            bbx = detect_by_mask(y_, 10)
            bbx_to_file[X.split("/")[-1].split(".")[0]] = bbx
            crop_x, _ = crop_volume(x, y, bbx)
            
            """ preprocessing for volume """
            x = clip(crop_x, -300, 500)
            if cfg.contrast_enhance:
                x = clahe_3d(x.astype(np.uint8), 0.0)
                x = 255 - x
            x = normalize(x)
            x = resize_volume(x, input_shape)
            # (h, w, d) - > (n, c, h, w, d)
            x = np.expand_dims(np.expand_dims(x, 0), 0)
            
            """ predict """ 
            x = torch.from_numpy(x).float().to(device)
            pred = model(x)
            # pred = F.softmax(pred, dim=1)
            """ post-processing: crfasrnn """
            pred = post_model(pred, x)
            pred = torch.argmax(pred, dim=1)[0].cpu().numpy().astype('float32')

            """ resize to the cropped shape """
            y_pred_reshape = resize_volume(pred, crop_x.shape, is_label=True)
            assert y_pred_reshape.shape == crop_x.shape, "excepted shape {}, got {}.".format(y.shape, y_pred_reshape.shape)
            
            """ to original size """
            y_pred = np.zeros(y.shape)
            y_pred[bbx[0]:bbx[1], bbx[2]:bbx[3], bbx[4]:bbx[5]] = y_pred_reshape
            
            """ post-processing: filter FP/small noise """
            y_pred_fixed = np.zeros(y.shape)
            y_pred_fixed = y_pred.copy()
            y_pred_fixed = filter_noise(y_pred_fixed)
            
            """ cal. for accuracy """
            y_pred = torch.from_numpy(y_pred).float().to(device)
            y_pred_fixed = torch.from_numpy(y_pred_fixed).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            acc = dice(y_pred, y).item()
            acc_fixed = dice(y_pred_fixed, y).item()
            test_acc += acc
            test_fixed_acc += acc_fixed
            print("{} | dice: {:.2f}, fixed: {:.2f}".format(X.split('/')[-1].split('.')[0], acc, acc_fixed))
            
            """ save as nii """
            y = y.cpu().numpy().astype('float32')
            y_pred_fixed = y_pred_fixed.cpu().numpy().astype('float32')
            save_result(y, os.path.join(save_path, "label_" + X.split('/')[-1]), affine=y_affine)
            save_result(pred, os.path.join(save_path, "pred_small_" + X.split('/')[-1]), affine=y_affine)
            save_result(y_pred_fixed, os.path.join(save_path, "result_" + X.split('/')[-1]), affine=y_affine)

    test_acc /= len(test_data.img_paths)
    test_fixed_acc /= len(test_data.img_paths)
    print("--------------\n[INFO] Avg. Dsc: {:.4f}, fixed Dsc: {:.4f}\n".format(test_acc, test_fixed_acc))
    
    save_bbx_path = "result/bbx.json"
    with open(save_bbx_path, "w") as outfile:
        json.dump(bbx_to_file, outfile)
    

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    main()