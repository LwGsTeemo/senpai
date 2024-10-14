import os
import time
import json
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.Checkpoint import load_model, load_ckpt, save_ckpt
from dataio.Dataset_v2 import ImageDataset
from utils.toolkit import get_loss_function, get_optimizer
from utils.metrics import MulticlassDiceScore
from utils.early_stop import EarlyStopper
from config import training_config as cfg
from utils.figure import plot_fig
from test_kfold import predict
from dataio.transform import train_transforms

def main():
    K = 5
    results = {}
    save_result_path = cfg.checkpoint_path + "results.json"
    if os.path.exists(save_result_path):
        with open(save_result_path, "r") as f:
            data = json.load(f)
        
    start = 1
    for i in range(start, K+1):
        start = i
        if not os.path.exists(cfg.checkpoint_path + f'K{i+1}/'):
            break
        if os.path.exists(save_result_path):
            results[i] = data[f'{i}']

    since = time.time()
    for fold in range(start, K+1):
        print(f'\n[INFO] K{fold} started.')
        
        root = cfg.dataset.root
        method = cfg.method
        cache_dir = cfg.dataset.cache_dir
        group = f'K{fold}/'
        
        data_path = {
            "train_img": root + group + cfg.dataset.train_img_path,
            "train_label": root + group + cfg.dataset.train_label_path,
            "test_img": root + group + cfg.dataset.test_img_path,
            "test_label": root + group + cfg.dataset.test_label_path
        }
        cache_path = {
            'train_img_cache': cache_dir + root + method + group + cfg.dataset.train_img_path,
            'train_label_cache': cache_dir + root + method + group + cfg.dataset.train_label_path, 
            'test_img_cache': cache_dir + root + method + group + cfg.dataset.test_img_path,
            'test_label_cache': cache_dir + root + method + group + cfg.dataset.test_label_path
        }
        
        input_shape = cfg.dataset.input_shape
        NUM_CLASSES = cfg.dataset.num_class
        BATCH_SIZE = cfg.batch_size
        EPOCHS = cfg.epoch
        method = cfg.method

        print("[INFO] Load data...")
        params = {
            'size': input_shape,
            'n_class': NUM_CLASSES
        }
        if os.path.exists(cache_path['train_img_cache']) and os.path.exists(cache_path['test_img_cache']):
            train_ds = ImageDataset(cache_path['train_img_cache'], cache_path['train_label_cache'], transform=train_transforms, **params)
            val_ds = ImageDataset(cache_path['test_img_cache'], cache_path['test_label_cache'], **params)
        else:
            train_ds = ImageDataset(data_path['train_img'], data_path['train_label'], transform=train_transforms, **params)
            train_ds.create_dataset(cache_path['train_img_cache'], cache_path['train_label_cache'])
            val_ds = ImageDataset(data_path['test_img'], data_path['test_label'], **params)
            val_ds.create_dataset(cache_path['test_img_cache'], cache_path['test_label_cache'])
        
        train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)

        print("--------------\n[INFO] Load model...")
        model = load_model(cfg.model_name, input_shape, NUM_CLASSES)
        print(f'[INFO] Model: {cfg.model_name}')
        if not cfg.training.model_parallel:
            model = model.to(device)
            if fold == 1:
                summary(model, (1,) + tuple(input_shape))
        
        loss_fn = get_loss_function(cfg.loss, device)
        print(cfg.loss)
        dice = MulticlassDiceScore()
        optimizer = get_optimizer(cfg, model.parameters())
        
        checkpoint_path = cfg.checkpoint_path + f'K{fold}/'
        model, init_epoch, best_acc, optimizer, counter_init = load_ckpt(model, checkpoint_path, optimizer)
        
        early_stopping = EarlyStopper(patience=cfg.patience, counter_init=counter_init, acc_init=best_acc, min_delta=1)
        
        train_acc_epoch, train_loss_epoch = [], []
        val_acc_epoch, val_loss_epoch = [], []
        print("--------------\n[INFO] Start Training...")
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(init_epoch, EPOCHS):
                """ training phase"""
                model.train()
                train_loss, train_acc = 0., 0.
                train_size = 0
                progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch+1}/{EPOCHS} ')
                for batch, (X, mask) in progress:
                    X, mask = X.to(device), mask.to(device)
                    pred_mask = model(X)
                    
                    optimizer.zero_grad()
                    if cfg.deep_supervision:
                        total_loss = 0
                        for out in pred_mask:
                            total_loss += loss_fn(out, mask)
                        total_loss.backward()
                    else:
                        loss = loss_fn(pred_mask, mask)
                        loss.backward()
                    
                    torch.cuda.synchronize()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    train_size += len(X)
                    if cfg.deep_supervision:
                        train_loss += total_loss.item()
                        train_acc += dice(pred_mask[0], mask).item()
                    else:
                        train_loss += loss.item()
                        train_acc += dice(pred_mask, mask).item()
                    progress.set_postfix({"loss": train_loss/train_size,"acc": train_acc/train_size})
                train_acc_epoch.append(train_acc/train_size)
                train_loss_epoch.append(train_loss/train_size)
                
                """ validation phase """
                model.eval()
                val_size = 0
                val_loss, val_acc = 0., 0.
                with torch.no_grad():
                    for X, mask in val_dataloader:
                        X, mask = X.to(device), mask.to(device)
                        if cfg.deep_supervision:
                            pred_mask = model(X)[0]
                        else:
                            pred_mask = model(X)

                        val_loss += loss_fn(pred_mask, mask).item()
                        val_acc += dice(pred_mask, mask).item()
                        val_size += len(X)
                
                val_acc /= val_size
                val_loss /= val_size                    
                print("val loss: {:.4f}, val acc: {:.4f}".format(val_loss, val_acc))
                val_acc_epoch.append(val_acc/val_size)
                val_loss_epoch.append(val_loss/val_size)
                
                if early_stopping.early_stop(val_acc):
                    print("--------------\n[INFO] EarlyStopped.")
                    break
                
                if val_acc > best_acc:
                    print("[INFO] val acc improved from {:.4f} to {:.4f}, saving the best model to {}.".format(best_acc, val_acc, checkpoint_path))
                    best_acc = val_acc
                    save_ckpt(epoch, model, optimizer, best_acc, checkpoint_path + "best.pth", early_stopping.counter)
                
                save_ckpt(epoch, model, optimizer, best_acc, checkpoint_path + "ckpt.pth", early_stopping.counter)
            
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
        print(f'[INFO] Test K{fold}...')
        model, _, _, _, _ = load_ckpt(model, checkpoint_path, optimizer=None, is_best=True)
        model.eval()
        _, test_acc = predict(root, input_shape, NUM_CLASSES, method, model, fold, data_path['test_img'], data_path['test_label'], device=device, **params)
        print("[INFO] {}-fold | Accuarcy: {:.4f}".format(fold, test_acc))
        results[fold] = 100.0 * test_acc
        
        with open(save_result_path, "w") as outfile:
            json.dump(results, outfile)
    
        """plot training history """
        plot_fig(train_acc_epoch, train_loss_epoch, val_acc_epoch, val_loss_epoch, f'K{fold}')
    
    
    print(f'\nK-FOLD CROSS VALIDATION RESULTS FOR {K} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')
    
    time_elapsed = time.time() - since
    print('\nTraining completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    

if __name__ == "__main__":
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    main()