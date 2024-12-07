# Mini-Project 10: Customized U-Net

## Team Members

- **Derek Zhang**
- **Yichi Zhang**
- **Yiling Hu**
- **Ziyang Liang**

---

## Project Overview

This project applies a **customized U-Net model** for road segmentation tasks using a dataset of urban images with ground truth masks. The U-Net model has been tailored to achieve high segmentation accuracy, particularly in identifying road areas from images. We optimized the model with hyperparameter tuning and additional techniques to improve performance and visualization.

## Setup Instructions

Ensure Python 3.x is installed on your system. Install the required libraries by running:

```bash
pip install tensorflow opencv-python-headless numpy matplotlib scikit-learn
```

## Dataset

### Dataset Preparation

The dataset is expected to be a zip file named `cityscape_dataset.zip` stored in the [Google Drive root directory](https://drive.google.com/drive/folders/1qRVz70ixvsj76Pp2Gwv982A6ZFzYPG-o).

- **Dataset Location**: `cityscape_dataset.zip` (downloaded and stored locally after unzipping).
- **Dataset Size**: 1391 images with corresponding ground truth masks.
- **Paths in Code**:
  - Images: `leftImg8bit/train`
  - Masks: `gtFine/train`

### Unzip the Dataset

In the Google Colab environment:

```python
!unzip -oq '/content/drive/MyDrive/cityscape_dataset.zip' -d '/content/'
```

## Model Architecture

You can find the training code in the same [Google Drive root directory](https://drive.google.com/drive/folders/1qRVz70ixvsj76Pp2Gwv982A6ZFzYPG-o) named **unet_training.py**.
The model is a U-Net architecture with:

- **Input Shape**: 256x256x3
- **Batch Size**: 8 (optimized for the input size)
- **Optimizer**: Adam with learning rate of 1e-4
- **Loss Function**: Combination of Binary Cross-Entropy and Dice Loss
- **Metrics**: IoU (Intersection over Union) and Accuracy

The U-Net includes data augmentation parameters:

- Rotation, width, and height shifts
- Zoom, brightness, and horizontal flip adjustments

## Training and Validation

- **Training/Validation Split**: The data is split into 80% for training and 20% for validation.
- **Batch Size**: 8 (to manage memory with 256x256 image inputs).
- **Epochs**: Trained for 50 epochs with early stopping and learning rate reduction on plateau.

### Augmentation Parameters

We used the following data augmentation for training:

- **Rotation Range**: 15 degrees
- **Width/Height Shift Range**: 10%
- **Zoom Range**: 20%
- **Horizontal Flip**: Enabled
- **Brightness Range**: 0.8 to 1.2

## Model Training

To train the model, run the following command:

```python
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=(X_val, y_val),
    epochs=15,
    callbacks=callbacks
)
```

### Callbacks

- **ModelCheckpoint**: Saves the best model: unet_best_model.keras.
- **EarlyStopping**: Stops training if no improvement in 10 epochs.
- **ReduceLROnPlateau**: Reduces learning rate by 50% if no improvement in 5 epochs.

## Evaluation

The model's performance was evaluated using IoU and accuracy metrics:

```python
test_loss, test_accuracy, test_iou = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
print(f'Test IoU: {test_iou}')
```

## Visualization of Results

The code provides functions to visualize the original images, ground truth masks, and predicted masks.
[Results of visualization](https://drive.google.com/drive/folders/1qRVz70ixvsj76Pp2Gwv982A6ZFzYPG-o)

1. **Display Sample Images and Masks**: Shows random samples of images and masks before training.
2. **Plot Training Metrics**: Plots training and validation loss and IoU across epochs.
3. **Display Predictions**: Generates side-by-side comparisons of images, ground truth masks, and predicted masks with IoU scores.

To display predictions with IoU:

```python
display_predictions(model, X_test, y_test)
```

## Files in Repository: [Google Drive Link](https://drive.google.com/drive/folders/1qRVz70ixvsj76Pp2Gwv982A6ZFzYPG-o)

- `cityscape_dataset.zip`: Compressed dataset for training and testing.
- `unet_best_model.keras`: Saved model with the best performance.
- `unet_training.py`: The Python script containing all steps for loading data, training, and evaluating the model.

## Results 

The model achieved the following results after **15 epochs** of training:
- **Accuracy**: 0.9494
- **IoU Metric**: 0.8698
- **Loss**: 0.1159
- **Validation Accuracy**: 0.9303
- **Validation IoU**: 0.8149
- **Validation Loss**: 0.2107

In [Google Drive Link](https://drive.google.com/drive/folders/1qRVz70ixvsj76Pp2Gwv982A6ZFzYPG-o), you can find the screenshots for the result, and log.

## Improvements

- **Hyperparameter Tuning**: While the current model achieved notable results with an accuracy of 0.9494 and an IoU of 0.8698, further optimization of hyperparameters (e.g., learning rate, batch size, and augmentation settings) could enhance these metrics. Additional experimentation with learning rate schedules or adaptive learning rate techniques may yield improved convergence.
  
- **Extended Training on the Full Dataset**: The model was trained on a subset of the Cityscapes dataset for faster training. Expanding the training set to the full dataset could improve generalization, likely boosting both validation accuracy and IoU.

- **Increased Epochs**: The model was trained for 15 epochs for demonstration purposes. Increasing the number of epochs, while monitoring for overfitting, could allow the model to learn more nuanced features and potentially improve accuracy and IoU on validation data.

- **Fine-Tuning Learning Rate**: Reducing the learning rate as training progresses, particularly after reaching initial plateaus, may help the model converge better and achieve a higher final accuracy. Implementing a more gradual learning rate decay, such as cosine annealing, could be beneficial.