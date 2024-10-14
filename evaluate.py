import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from Checkpoint import make_or_restore_model
# from Dataset import ImageDataset
from dataio.Dataset_v2 import ImageDataset
from metrics import MulticlassDiceScore
from toolkit import get_loss_function
from config import training_config as cfg

def main():
    root = cfg.dataset.root
    test_img_path = cfg.dataset.test_img_path
    test_label_path = cfg.dataset.test_label_path

    cache_path = {
        'test_img_cache': cfg.dataset.test_img_cache,
        'test_label_cache': cfg.dataset.test_label_cache,
    }

    input_shape = cfg.dataset.input_shape
    NUM_CLASSES = cfg.dataset.num_class
    BATCH_SIZE = cfg.batch_size
    method = cfg.method

    print("[INFO] Load test data...")
    params = {
        'size': input_shape,
        'n_class': NUM_CLASSES
    }
    if os.path.exists(cache_path['test_img_cache']):
        test_data = ImageDataset(cache_path["test_img_cache"], cache_path["test_label_cache"], **params)
    else:
        test_data = ImageDataset(root + test_img_path, root + test_label_path, **params)
        test_data.create_test_data(root, method)
    
    test_ds = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)
    
    print("[INFO] Load model...")
    checkpoint_path = cfg.checkpoint_path
    
    model, _ = make_or_restore_model(cfg.model_name, input_shape, NUM_CLASSES, checkpoint_path)
    if not cfg.training.model_parallel:
        model = model.to(device)
    
    loss_fn = get_loss_function(cfg.loss, device)
    dice = MulticlassDiceScore()
    
    print("--------------\n[INFO] Evaluation...")
    model.eval()
    test_loss, test_acc = 0, 0
    data_size = 0
    with torch.no_grad():
        progress = tqdm(enumerate(test_ds), total=len(test_ds))
        for batch, (X, y) in progress:
            X, y = X.to(device), y.to(device)
            if cfg.deep_supervision:
                pred = model(X)[0]
            else:
                pred = model(X)

            test_loss += loss_fn(pred, y).item()
            acc = dice(pred, y).item() * len(X)
            data_size += len(X)
            test_acc += acc
            progress.set_postfix({"loss": test_loss/data_size,"acc": test_acc/data_size})
    test_loss /= data_size
    test_acc /= data_size
    print("[INFO] Test loss: {:.4f}, Test acc: {:.4f}\n".format(test_loss, test_acc))


if __name__ == "__main__":
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    main()