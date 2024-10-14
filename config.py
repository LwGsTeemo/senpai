from easydict import EasyDict
from utils.utils import CONFIGS, K5_pancreas_dataset,pancreas_dataset

training_config = pancreas_dataset(CONFIGS['UNET_CBAM_pool_DS'])

training_config.dataset.input_shape = (192, 128, 128)
training_config.dataset.resample_slice = (1.0, 1.0, 1.0)

training_config.batch_size = 1
training_config.epoch = 500
training_config.patience = 500

training_config.method = "local_resample_clahe_big/"
training_config.checkpoint_path = "saved_model/" + training_config.dataset.name + training_config.method + training_config.model_name + "/"

training_config.dataset.contrast_enhance = True
training_config.dataset.cache_dir = "cache/"

training_config.training = EasyDict()
training_config.training.data_parallel = False
training_config.training.model_parallel = False

training_config.predict = EasyDict()
training_config.predict.resample_back = True
training_config.predict.crf = False
training_config.predict.smoothing = False
training_config.predict.filter_noise = False
