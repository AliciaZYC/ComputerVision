# Mini-Project3: Webcam Transformation

###  Project Group 4

- Derek Zhang
- Yichi Zhang
- Yiling Hu
- Ziyang Liang


## Project Description

This group project implements a video capture application using OpenCV. It allows real-time image warping effects such as **translation**, **rotation**, **scaling**, and **perspective transformation**. The user can switch between effects using keyboard controls, and the frame rate is calculated to ensure smooth performance.
This application is a simple webcam-based tool that allows users to apply real-time transformations such as translation, rotation, scaling, and perspective to the video feed from their webcam. The app is built using OpenCV and Python.

## Features

- **Real-time Video Feed**: The application captures a live video feed from the user's webcam and allows transformations on the fly.
- **Transformation Modes**:
  - **Translation**: Shift the video frame horizontally or vertically.
  - **Rotation**: Rotate the video frame by a specified degree.
  - **Scaling**: Resize the video frame by a scaling factor.
  - **Perspective**: Apply a perspective transformation to simulate a different viewpoint.
- **Toggle Between Modes**: Switch between different transformation modes using keyboard input(pressing the same key again will return to the original mode):
  - `t`: Translation
  - `r`: Rotation
  - `s`: Scaling
  - `p`: Perspective
- **Original and Transformed Views**: The application shows both the original frame and the transformed frame side by side.
- **FPS Display**: The frames per second (FPS) is displayed on both the original and transformed frames.


## Setup Instructions

1. **Install Dependencies:**
    - You need Python 3.x and OpenCV.
    - Install OpenCV via pip:
        
        ```bash
        pip3 install opencv-python
        ```
        
2. **Running the Application:**
    - Run the script with:
        
        ```bash
        python WebCam2.py
        ```

## Usage Guide

- The application captures video from your webcam and allows you to apply different warping effects in real time.
- **Controls:**
    - Press `t` to apply/cancel translation.
    - Press `r` to apply/cancel rotation.
    - Press `s` to apply/cancel scaling.
    - Press `p` to apply/cancel perspective transformation.
    - Press `q` to quit the application.


### Demonstrating Video

url: https://drive.google.com/drive/folders/10ehLSJHL3uk9QPhVTZnkJCsxqea7fR6Z?usp=share_link
