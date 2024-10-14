import os
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from Checkpoint import make_or_restore_model
# from Dataset import ImageDataset
from dataio.Dataset_v2 import ImageDataset
from toolkit import get_loss_function, get_optimizer
from metrics import MulticlassDiceScore
from early_stop import EarlyStopper
from config import training_config as cfg
from figure import plot_fig

def main():
    root = cfg.dataset.root
    train_img_path = cfg.dataset.train_img_path
    train_label_path = cfg.dataset.train_label_path
    val_img_path = cfg.dataset.val_img_path
    val_label_path = cfg.dataset.val_label_path
    
    cache_path = {
        'train_img_cache': cfg.dataset.train_img_cache,
        'train_label_cache': cfg.dataset.train_label_cache,
        'val_img_cache': cfg.dataset.val_img_cache,
        'val_label_cache': cfg.dataset.val_label_cache,
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
    if os.path.exists(cache_path['train_img_cache']) and os.path.exists(cache_path['val_img_cache']):
        train_ds = ImageDataset(cache_path['train_img_cache'], cache_path['train_label_cache'], **params)
        val_ds = ImageDataset(cache_path['val_img_cache'], cache_path['val_label_cache'], **params)
    else:
        train_ds = ImageDataset(root + train_img_path, root + train_label_path, **params)
        train_ds.create_dataset(root, method)
        val_ds = ImageDataset(root + val_img_path, root + val_label_path, **params)
        val_ds.create_dataset(root, method)
    
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)

    print("--------------\n[INFO] Load model...")
    checkpoint_path = cfg.checkpoint_path
    model, init_epoch = make_or_restore_model(cfg.model_name, input_shape, NUM_CLASSES, checkpoint_path)
    print(f'[INFO] Model: {cfg.model_name}')
    if not cfg.training.model_parallel:
        model = model.to(device)
        summary(model.to(device), (1,) + tuple(input_shape))
    
    loss_fn = get_loss_function(cfg.loss, device)
    dice = MulticlassDiceScore()
    optimizer = get_optimizer(cfg, model.parameters())
    early_stopping = EarlyStopper(patience=cfg.patience, min_delta=1)
    
    best_acc = 0.
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
            val_acc_epoch.append(val_acc)
            val_loss_epoch.append(val_loss)
            print("val loss: {:.4f}, val acc: {:.4f}".format(val_loss, val_acc))
            
            if early_stopping.early_stop(val_acc):
                print("--------------\n[INFO] EarlyStopped.")
                break
            
            if val_acc > best_acc:
                print("[INFO] val acc improved from {:.4f} to {:.4f}, saving the best model to {}.".format(best_acc, val_acc, checkpoint_path))
                best_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path + "ckpt_" + str(epoch + 1).rjust(4, '0') + ".pth")
        
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    """plot training history """
    plot_fig(train_acc_epoch, train_loss_epoch, val_acc_epoch, val_loss_epoch, "result")
    

if __name__ == "__main__":
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    main()