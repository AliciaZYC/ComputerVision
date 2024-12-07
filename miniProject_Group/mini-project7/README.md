# Mini-Project 7: Live Classifier

## Team Members

- **Derek Zhang**
- **Yichi Zhang**
- **Yiling Hu**
- **Ziyang Liang**

---

## Project Overview

The **Live Classifier** is a real-time object classification system that identifies objects from a live video feed using a **Convolutional Neural Network (CNN)**. The system processes each video frame and classifies it into one of the predefined categories: remote control, cell phone, TV, and an additional custom class. This project demonstrates the seamless integration of a trained CNN model with **OpenCV** to classify objects with real-time visual feedback.

---

## Features

- **Real-time Object Classification:**
  - Classifies each frame in the webcam feed with high accuracy and minimal latency.
  - Predefined categories include remote control, cell phone, TV, and a custom class.
- **Live Feed Integration:**
  - Uses OpenCV to capture frames from a connected webcam.
- **Confidence Overlay:**
  - Displays the predicted class label and confidence score directly on the video feed.
- **Modular CNN Architecture:**
  - Easy to adjust and retrain with different datasets.

---

## Setup Instructions

### 1. Install Dependencies

Ensure Python 3.x is installed on your system. Install the required libraries by running:

```bash
pip install torch torchvision numpy tensorflow

```

### 2. Dataset Preparation

- We collect **100 images per class** for the following categories:
  - Remote Control
  - Cell Phone
  - TV
  - Custom class: Lamp
- We organize the dataset with the following structure, and adjust the images to have consistent size (3 _ 64 _ 64) and format, suitable for input to the CNN. The train dataset contains **80 images each class** and the test dataset contains **20 images each class**.

```bash
dataset/
├── train/
│   ├── TV/
│   ├── MobilePhone/
│   ├── RemoteControl/
│   └── Lamp/
└── test/
    ├── TV/
    ├── MobilePhone/
    ├── RemoteControl/
    └── Lamp/
```

- Source for dataset:

  - Remote Control and Lamp class is extracted from tiny-ImageNet-200.
  - Cell Phone and TV class are downloaded from google using Beautifulsoup.

- In order to load our dataset, you can either download from this Github repository, or with the following link, both ways need to unzip file in the same project folder to ensure that the code can
  run smoothly:
  [Google Drive Link](https://drive.google.com/file/d/1RsYVyk6gLYVfsw252bGxhKlMFoQ_Bc4d/view?usp=sharing)

---

## Running the Application

### 1. Train the CNN Model

Use the `train.py` script to train the CNN on your dataset:

```bash
python3 train.py

```

This script:

- Loads the dataset, preprocesses images, and trains the model.
- Saves the best-performing model for use with the live classifier.

### 2. Run the Live Classifier

Launch the live object classifier using the `Webcam.py` script:

- For live video capuring and classification:

```bash
python3 Webcam.py

```

- For uploaded video test classification:

```bash
python3 Webcam.py -f name_of_test_video.mp4 -o output_video.avi

```

**Controls:**

- **`q`**: Quit the application
- **`s`**: Pause/Resume the classifier

---

## CNN Model Architecture

The CNN model includes:

- **Convolutional Layers:** Extract features from input images.
- **Pooling Layers:** Reduce dimensionality and improve efficiency.
- **Dense Layers:** Fully connected layers for classification.
- **Activation Functions:** Use ReLU for hidden layers and Softmax for the output layer.

---

## Output

- **Real-Time Classification Display:**
  - The live feed shows the predicted object class and the confidence score on the screen.
- **Video Demonstration:**
  - A recorded video demonstrating the system in action is available in the GitHub repository or Google Drive.

---

## Demonstration Videos

Sample videos showing the classifier in action are uploaded onto the repository or with google drive:

- **Recorded test video:**

  - `test_video.mp4` → `output_video.avi`
  - `test.mp4` → `output_video1.avi`
  - [Google Drive Link](https://drive.google.com/file/d/1BlGyQMHsTVs61cnC5Zq2G5uuyOFMi2Wc/view?usp=sharing)

---
