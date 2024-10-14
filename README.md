# Biliary_and_Pancreatic_Segmentation_Pytroch

This is the implementation of this [Master's thesis/project](https://etd.lib.nycu.edu.tw/cgi-bin/gs32/ncugsweb.cgi?o=dncucdr&s=id=%22GC111522081%22.&searchmode=basic). If you need a reference please see [citation](#citation).

## Pre-Requisites
1. Open anaconda and activate your environment.
	```bash
	conda activate your_env
	```
2. Open the directory.
	```bash
	cd ./Biliary_and_Pancreatic_Segmentation_Pytroch/
	```
3. Install all the required packages by type this command below:
	```bash
	pip install -r requirements.txt
	```
4. Download all the data. Follow the [steps](#dataset-collection) as below.


## Dataset collection
1. This project use TCIA Pancreas-CT dataset for experiments. [Download here](https://www.cancerimagingarchive.net/collection/pancreas-ct/).
    > :warning: 
    > Download this dataset needs NBIA Data Retriever.
2. After download completed, convert the image from dicom (.dcm) datatype to nifti (.nii) using `dicom2nifti`.
    Sample code is showed below: 
    ```python
    import dicom2nifti
    import os

    path_to_all_patients = "dataset/PANCREAS-CT/images_dcm"

    patient_folders = os.listdir(path_to_all_patients)
    patient_folders.sort()

    """ transform per person """
    for p in patient_folders:
        print("process on {}".format(p))
        dicom2nifti.convert_directory(os.path.join(path_to_all_patients, p), os.path.join(path_to_all_patients, p), reorient=False)
    ```
3. To split the dataset into train-test or k-fold, use `data_utils.py`.
4. Place the split dataset on the same path with your `train.py` or `train_kfold.py`.

## Usage
- train the model (k-fold)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python train_kfold.py
    ```
- predict (k-fold)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python predict_v2.py
    ```

## Code explain
- `dataio/`: Code which is related to handling the dataset are placed here.
    - `transform.py`: Do transform (rescale, rotate, resample, etc...) if needed.
    - `Dataset.py`: (Resize version) Read the dataset, do pre-processing, and save the preprocessed images as .npy file.
    - `Dataset_v2.py`: (Resample version) Read the dataset, do pre-processing, and save the preprocessed images as .npy file.
- `losses/`: Loss functions that might used in experiments are stored here, e.g. dice loss, ms-ssim, cross entropy, etc.
- `models/`: Models are all here, it's too much so I am not going to list them here. 
- `utils/`: Some useful functions or classes are placed here, e.g. early stop, loss function, metric, preprocessing steps, etc.
    - `Checkpoint.py`: Used to save/load chekpoint and load models.
    - `early_stop.py`: Early stop criteria. To prevent from overfitting.
    - `figure.py`: To plot the figure for the training/test curve (accuracy/loss).
    - `Loss.py`: The loss function used in this project is placed here, combining two or the above types of loss functions; the code of each loss function is in `losses/`.
    - `metrics.py`: Score criteria is placed here, e.g. Dice Similarity Coefficient (DSC).
    - `postprocessing.py`: All the post-processing methods.
    - `preprocessing.py`: All the pre-processing methods.
    - `toolkit.py`: Some useful tools are placed here, e.g. save cache, load path, get loss function, etc.
    - `utils.py`: Record the settings for the models and the datasets. Use `CONFIGS` in `config.py` and change the model name to switch to the model/dataset you need.
- `data_utils.py`: Used to split dataset into train-test or k-fold.
- `config.py`: The parameters and settings for training or testing. Change the setting here instead of opening the main file.
- `evaluate.py`: Evaluate testing data without resizing images back to their original size.
- `train.py`: Train the model with training data, and save the model weights.
- `predict.py`: (Resize version) Predict and evaluate testing data by resizing images to their original size and saving the results.
- `predict_v2.py`: (Resample version) Predict and evaluate testing data by resizing images to their original size and saving the results.
- `test_kfold.py`: The function `predict(...)` here is used in `train_kfold.py`, which is used to predict/evaluate the test data for the K-fold method.
- `train_kfold.py`: Train the model with training data in the K-fold method,  save the model weights in each fold, and summarize the accuracy. (Main function)

## References
1. Roth, H., Farag, A., Turkbey, E. B., Lu, L., Liu, J., & Summers, R. M. (2016). Data From Pancreas-CT (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU

> Other reference are all list in the paper. [see here](https://etd.lib.nycu.edu.tw/cgi-bin/gs32/ncugsweb.cgi?o=dncucdr&s=id=%22GC111522081%22.&searchmode=basic)

## Citation
```
M.-Y. Hsiao, The Development of Deep Learning-based Pancreas Segmentation Methods, Master’s thesis, 2024.
```