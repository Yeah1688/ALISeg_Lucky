# USER_GUIDE.md

This guide provides a detailed overview of the project structure, data preparation requirements, and key configuration parameters for training, prediction, and evaluation using the model.

---

## 1. Directory Structure

The project follows a specific organizational layout within the `ALISeg/` root directory:

* **`VOCdevkit/`**: The main directory for VOC format datasets.
    * `VOC2007/ImageSets/Segmentation/`: Files for dataset splitting.
    * `VOC2007/JPEGImages/`: Original input images.
    * `VOC2007/SegmentationClass/`: Annotation mask images.
    * `VOC2007/Muddy_Predict/`: Images awaiting prediction.
    * `VOC2007/JPEGImages_predict/`: Directory for prediction result images.
* **`logs_muddy/`**: Directory for saving training weights and logs.
* **`miout_muddy/`**: Directory for mIoU evaluation results.
* **`nets/`**: Contains model definitions.
* **`utils/`**: Contains utility functions.
* **`train_MDCH.py`**: Script used for training the model.
* **`predict_MDCH.py`**: Script used for running predictions.
* **`get_miou_MDCH.py`**: Script used for model evaluation.
* **`json_to_dataset.py`**: Script for converting LabelMe annotations.

---

## 2. Data Preparation

### (1)Input Images (`JPEGImages`) 
* **Formats**: `.jpg` `.jpeg` `.png` `.tif` `.bmp`.
* **Color Space**: RGB (automatic conversion).
* **Dimensions**: Any size (automatically resized to `input_shape`).

### (2)Annotation Images (`SegmentationClass`) 
* **Format**: `.png`.
* **Type**: Grayscale or 8-bit color images.
* **Pixel Values**: Integer values at each pixel represent the class index.

---

## 3. Key Configuration Parameters

### Training Configuration (`train.py`) 
* **`Cuda`**: `True` (whether to use CUDA).
* **`seed`**: `11` (random seed to fix experimental results).
* **`num_classes`**: `2` (number of classes).
* **`input_shape`**: `[256, 256]` (input image dimensions).
* **`downsample_factor`**: `16` (downsampling factor: 8 or 16)
* **`save_dir`**: `'logs_muddy/MDCH'` (directory for saving weights).

### Prediction Configuration 
In `deeplab_MDCH`:
* **`model_path`**: `'logs_muddy/MDCH_111/best_epoch_weights.pth'` (weight path)
* **`num_classes`**: `2` (number of classes).
* **`backbone`**: `"mobilevit"` (backbone network).
* **`input_shape`**: `[256, 256]` (input size).
* **`downsample_factor`**: `16` (downsampling factor)
* **`mix_type`**: `1` (visualization mode).
* **`cuda`**: `True` (whether to use GPU).

In `predict_MDCH`
* **`mode`**: `"dir_predict"` (batch image prediction).
* **`dir_origin_path`**: `"VOCdevkit/VOC2007/Muddy_Predict"` (path for target images).
* **`dir_save_path`**: `"VOCdevkit/VOC2007/JPEGImages_predict/MDCH"` (path for saving results).

### Evaluation Configuration 
In `deeplab_MDCH`:
* Configuration parameters match the prediction settings (e.g., `model_path`, `backbone`, `cuda`, etc.).

In `get_miou_MDCH.py`:
* **`miou_mode`**: `0` (0: full process, 1: prediction only, 2: calculation only).
* **`num_classes`**: `2` (number of classes).
* **`name_classes`**: `["water", "land"]` (class names).
* **`VOCdevkit_path`**: `'VOCdevkit'` (dataset path).
* **`miou_out_path`**: `"miout_muddy/MDCH"` (path for evaluation results).
