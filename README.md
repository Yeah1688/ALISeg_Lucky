## Download
Click on the green button "clone or download". Then, unzip it on your computer.

## 0. Requirements
    torch>=1.7.1
    torchvision>=0.8.2
    Pillow>=8.0.0
    opencv-python>=4.5.0
    opencv-contrib-python>=4.5.0
    numpy>=1.19.0
    scipy>=1.5.0
    matplotlib>=3.3.0
    tqdm>=4.50.0
    tensorboard>=2.4.0
    ptflops>=0.6.0
    onnx>=1.8.0
    onnx-simplifier>=0.3.0
    labelme>=5.0.0

You can install dependencies through the following command:
    pip install -r requirements.txt


## 1. Training Steps

Follow these steps to train the model using the **VOC dataset format**:

1.  **Format**: This project uses the VOC format for training.
2.  **Labels**: Place your label files (segmentation masks) in the following directory:
    `VOCdevkit/VOC2007/SegmentationClass`
3.  **Images**: Place your original image files in the following directory:
    `VOCdevkit/VOC2007/JPEGImages`
4.  **Annotation**: Run the `voc_annotation.py` script to generate the necessary `.txt` files for training.
5.  **Weight Path**: In `train_MDCH.py`, specify the directory path where you want to save the trained weights.
6.  **Execution**: Run `train_MDCH.py` to start the training process.

---

## 2. Evaluation Steps

To evaluate the performance of the trained model:

1.  **Model Path**: In `deeplab_MDCH.py`, update the `model_path` variable to point to your trained weight file.
    > **Note**: The weight file `logs_muddy/MDCH_111/best_epoch_weights.pth` is already provided and can be used directly for evaluation.
2.  **Output Path**: In `get_miou_MDCH.py`, modify `miou_out_path` to specify where the evaluation results should be saved.
3.  **Execution**: Run `get_miou_MDCH.py` to calculate metrics (e.g., mIoU).

---

## 3. Prediction Steps

To perform inference on new images:

1.  **Model Path**: In `deeplab_MDCH.py`, ensure the `model_path` points to the correct weight file.
    > **Note**: The weight file `logs_muddy/MDCH_111/best_epoch_weights.pth` is available for immediate use.
2.  **Directory Configuration**: In `predict_MDCH.py`, configure the following:
    * Set `dir_origin_path` to the folder containing your target images.
    * Set `dir_save_path` to the folder where prediction results will be stored.
3.  **Execution**: Run `predict_MDCH.py` to start the inference process.

