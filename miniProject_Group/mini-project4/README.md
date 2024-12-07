# Mini-Project4: Sewing Machine

###  Project Group 4

- Derek Zhang
- Yichi Zhang
- Yiling Hu
- Ziyang Liang

## Project Description
This project implements **ORB** (Oriented FAST and Rotated BRIEF), a fast and efficient alternative to SIFT, to detect and match features between live video feeds from two cameras. The project includes real-time visualization of feature detection and matching, along with a matching score that quantifies the quality of matches between the two frames. Additionally, it showcases how to draw lines connecting the matched feature points and display the frames per second (FPS) during the live operation.

## Features
- **Live webcam video feeds**: Captures real-time video from two camera inputs and processes them for feature detection and matching.
- **Feature detection and description**: Uses ORB to detect keypoints and compute descriptors in each frame from both cameras.
- **Feature matching**: Implements a **Brute-Force Matcher** to match ORB descriptors between frames.
- **Visualization of matched features**:
    - Draws lines connecting the matched feature points between the two camera frames.
    - Displays the matching points with clear visual lines and indicates their matching score.
- **Display matching score**: Quantifies and displays the match quality using the distance between the feature points. The shorter the distance, the better the match.
- **FPS display**: Shows the real-time frames per second for performance evaluation.


## Set Up Instruction

1. **Install Dependencies:**
    - You need Python 3.x and OpenCV.
    - Install OpenCV via pip:
        
        ```bash
        pip3 install opencv-python
        ```
        
2. **Running the Application:**
    - Run the script with:
        
        ```bash
        python webcam.py
        ```
3. **Move the testing camera to see the matching result**

## Usage Guide

- **Ensure your computer has two connected cameras**, as the project requires video input from two camera feeds (it can be one internal and 
one external camera). Once the script runs, the live feed from both cameras will appear, showing matched feature points between the two frames in real time, along with their matching scores and the FPS rate.
- **Corols:**
    - Press `q` to quit the application.

### Demonstrating Video

url: 
- Sample videos (1MB) attached in the mini-project4 Directory
- Demonstrating video [https://youtu.be/SOEiYZ-yjBM]()
