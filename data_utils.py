import os
import shutil
import random
import numpy as np
from sklearn.model_selection import KFold

from utils.toolkit import check_file_exists, load_path

def copy_file(path, new_path):
    dir = os.path.dirname(new_path)
    check_file_exists(dir)
    shutil.copyfile(path, new_path)

def split_into_k_folds(img_path, label_path, K):
    img_paths = load_path(img_path)
    label_paths = load_path(label_path)
    data_dict = [{'image': img, 'label': label} for img, label in zip(img_paths, label_paths)]
    
    # folder = KFold(n_splits=K, random_state=42, shuffle=True)
    folder = KFold(n_splits=K, random_state=None, shuffle=False)
    train_files, test_files = [], []
    for k, (idxTr, idxTs) in enumerate(folder.split(data_dict)):
        train_files.append(np.array(data_dict)[idxTr].tolist())
        test_files.append(np.array(data_dict)[idxTs].tolist())

    root = os.path.dirname(img_path).split("/")[-1]
    save_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), root + f"_{K}fold")
    for k in range(len(train_files)):
        dir_path = f"K{k+1}"
        train_path = os.path.join(save_dir, dir_path, "train")
        test_path = os.path.join(save_dir, dir_path, "test")
        for i in range(len(train_files[k])):
            copy_file(train_files[k][i]['image'], os.path.join(train_path, "image", train_files[k][i]['image'].split('/')[-1]))
            copy_file(train_files[k][i]['label'], os.path.join(train_path, "label", train_files[k][i]['label'].split('/')[-1]))
        for i in range(len(test_files[k])):
            copy_file(test_files[k][i]['image'], os.path.join(test_path, "image", test_files[k][i]['image'].split('/')[-1]))
            copy_file(test_files[k][i]['label'], os.path.join(test_path, "label", test_files[k][i]['label'].split('/')[-1]))

def divide_dataset(img_path, label_path):
    img_paths = load_path(img_path)
    label_paths = load_path(label_path)
    data_dict = [{'image': img, 'label': label} for img, label in zip(img_paths, label_paths)]
    
    random.shuffle(data_dict)
    train_percent = 0.8
    trainval_percent = 1.0

    num_train_val = int(len(data_dict) * trainval_percent)
    train_val = [i for i in range(num_train_val)]
    num_train = int(len(data_dict) * train_percent)
    val = [i for i in range(0, num_train, 8)]

    root = os.path.dirname(img_path)
    train_path = os.path.join(root, "train")
    val_path = os.path.join(root, "val")
    test_path = os.path.join(root, "test")

    for i in range(len(data_dict)):
        if i in train_val:
            if i in val: # val
                copy_file(data_dict[i]['image'], os.path.join(val_path, "image", data_dict[i]['image'].split('/')[-1]))
                copy_file(data_dict[i]['label'], os.path.join(val_path, "label", data_dict[i]['label'].split('/')[-1]))
            else: # train
                copy_file(data_dict[i]['image'], os.path.join(train_path, "image", data_dict[i]['image'].split('/')[-1]))
                copy_file(data_dict[i]['label'], os.path.join(train_path, "label", data_dict[i]['label'].split('/')[-1]))
        else: # test
            copy_file(data_dict[i]['image'], os.path.join(test_path, "image", data_dict[i]['image'].split('/')[-1]))
            copy_file(data_dict[i]['label'], os.path.join(test_path, "label", data_dict[i]['label'].split('/')[-1]))

if __name__ == "__main__":
    img_dir = "./dataset/Task07_Pancreas/image"
    label_dir = "./dataset/Task07_Pancreas/label"
    
    """ split into train-val-test (8:2) """
    # divide_dataset(img_dir, label_dir)
    
    """ split into k-fold (train-test) """
    k = 5
    split_into_k_folds(img_dir, label_dir, k)