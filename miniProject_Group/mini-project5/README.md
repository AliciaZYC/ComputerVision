# Mini-Project5: Panorama Creator

### Project Group 4

- Derek Zhang
- Yichi Zhang
- Yiling Hu
- Ziyang Liang

## Project Description

This project implements a **Panorama Creator** using **ORB** (Oriented FAST and Rotated BRIEF) to detect and match features between captured frames from a webcam. The project captures a series of images as the user **moves the camera horizontally**, detects features within these images, matches these features across frames, and stitches the images together to create a seamless panorama. Additionally, it displays the Frames Per Second (FPS) for performance evaluation and allows users to save the generated panorama image.

## Features

- **Live webcam video feed**: Captures real-time video from a webcam for the panorama creation process.
- **Frame capturing for panorama**:
  - Start capturing frames when the user presses the 's' key.
  - Stop capturing frames when the user presses the 'a' key.
- **Feature detection and description**: Utilizes ORB to detect keypoints and compute descriptors for each captured frame.
- **Feature matching and image stitching**: Matches ORB descriptors between consecutive frames, aligns frames based on matched features, and stitches them into a seamless panorama.
- **Display and save options**:
  - Displays the FPS rate to indicate the application’s performance.
  - Provides an option to save the generated panorama image to disk.

## Set Up Instructions

1. **Install Dependencies:**
   - You need Python 3.x and OpenCV.
   - Install OpenCV via pip:
     ```bash
     pip install opencv-python
     ```
2. **Running the Application:**
   - Run the script with:
     ```bash
     python WebCam.py
     ```

## Usage Guide

- **Controls**:
  - Press `s` to start capturing frames for the panorama.
  - Press `a` to stop capturing frames and generate the panorama.
  - Press `q` to quit the application.
- **Ensure your computer has a connected webcam**. Once the script runs, a live feed from the webcam will appear. The application will capture frames as you move the camera horizontally to capture different angles for the panorama. When ready, the software will stitch these frames together, display the panorama, and provide an option to save it.

## Code Overview

- **`calculate_fps`**: Calculates the frames per second (FPS) over a specified time interval for performance monitoring. Displays the FPS rate on the live video feed.
- **`stitch_images`**: Uses ORB (Oriented FAST and Rotated BRIEF) for feature detection and the BFMatcher to find feature matches between frames. This function calculates the homography matrix using RANSAC to align consecutive frames and stitches them together, creating a panorama. A width limit prevents errors from large image sizes.
- **`save_panorama`**: Displays the stitched panorama and saves it as an image file (`panorama.jpg`) if the stitching process is successful.
- **`main`**:
  - Initializes the webcam and captures frames on the press of the `s` key.
  - Stop capturing frames when the `a` key is pressed.
  - On pressing `a`, the script initiates the stitching process, displays the panorama, and allows the user to save the result.

### Demonstrating Video

- Sample video (1MB): sample_video1.mp4 in project folder
- Demonstrating video: https://drive.google.com/file/d/12XjrylNMOLbzikOrA2vOi8iOcvGxn3E6/view?usp=drive_link
