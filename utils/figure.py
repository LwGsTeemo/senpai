import os
import matplotlib.pyplot as plt
from datetime import datetime

from utils.toolkit import check_file_exists
from config import training_config as cfg

def plot_fig(train_acc, train_loss, val_acc, val_loss, save_name):
    plt.figure(figsize=(20, 5))
    epochs = range(1, len(train_acc) + 1)
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='accuracy')
    plt.plot(epochs, train_loss, label='loss')
    ax1.legend(loc="upper right")
    ax1.title.set_text('training')
    plt.xlabel('epoch')

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, label='accuracy')
    plt.plot(epochs, val_loss, label='loss')
    ax2.legend(loc="upper right")
    ax2.title.set_text('validation')
    plt.xlabel('epoch')

    dir_name = "./history"
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = cfg.model_name + "/" + save_name + "_" + current_datetime + ".png"
    check_file_exists(os.path.dirname(os.path.join(dir_name, save_name)))
    plt.savefig(os.path.join(dir_name, save_name))