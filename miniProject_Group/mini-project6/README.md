# Mini-Project6: Lane Detector

### Project Group 4

- Derek Zhang
- Yichi Zhang
- Yiling Hu
- Ziyang Liang

## Project Description

This project implements a **Lane Detector** using advanced image processing techniques to detect lane lines on the road. The system processes video frames to identify lane boundaries, enabling the detection of the left and right lane lines. The project uses **color filtering**, **edge detection**, and **Hough Transform** to accurately identify lane lines, even when they are partially obscured or faded.

## Features

- **Video file or live webcam feed**: Supports both pre-recorded video files and live feed from a connected camera for lane detection.
- **Real-time lane detection**:
  - Automatically processes each frame to detect lane lines in real time.
  - Utilizes color filtering to enhance the detection of lane lines on different road conditions.
- **Robust edge detection**: Uses Canny edge detection to identify significant changes in pixel intensity, helping to isolate lane lines.
- **Lane enhancement**: Applies morphological operations to strengthen the detection of dashed white lines by connecting them.
- **Hough Transform for line detection**: Uses the Hough Line Transform to detect and draw the left and right lane lines separately.
- **RANSAC to identify the best intersection**: Find the lines that best converge at a common point, helping detect lanes reliably.
- **Output visualization**:
  - Displays the lane detection results on the video in real time.
  - Saves the processed video with detected lane lines for later review.

## Set Up Instructions

1. **Install Dependencies:**
   - You need Python 3.x and OpenCV.
   - Install OpenCV via pip:
     ```bash
     pip install opencv-python numpy
     ```
2. **Running the Application:**
   - Run the script with a video file or live webcam:
     ```bash
     python WebCamSave.py -f video_file_name -o out_video.avi
     ```

## Usage Guide

- **Controls**:
  - The program automatically starts processing frames as soon as it runs.
  - Press `s` to Pause/Resume the application.
  - Press `q` to Quit the application and stop the lane detection.
- **Ensure your computer has a connected webcam or provide a valid video file**. The application processes each video frame in real time, applying image enhancement techniques to improve the detection of lane lines, even in challenging lighting conditions or with dashed lines.

## Code Overview

- **`color_filter`**: Filters the image to isolate the white lane lines, improving the accuracy of lane detection by reducing noise and irrelevant elements.
- **`region_of_interest`**: Defines a polygonal region of interest to focus the detection on the area where lane lines are most likely to appear.
- **`line_filter_pipeline`**: This function processes detected lines by:
   - Filtering out lines based on their slope (to remove horizontal or irrelevant lines).
   - Applying RANSAC to remove outlier lines and select the best lane lines.
   - Smoothing the detected lines by averaging over a buffer that stores detected lines from previous frames (helpful in real-time systems).
- **`line_slope_filter`**: Filters out lines that are too horizontal or vertical by calculating the slope. Lines with slopes between `-0.3` and `0.3` are excluded, as they are either too horizontal or not useful for lane detection.
- **`RANSAC`**: Applies the RANSAC algorithm to identify the best intersection of lines. It randomly selects pairs of lines, calculates their intersection, and checks how many other lines are close to this intersection (inliers). The goal is to find the lines that best converge at a common point, which can help detect lanes reliably.
- **`find_best_match_line`**: Separates the lines into left and right lane candidates based on slope, then identifies the best matches by comparing slopes and distances between lines. It returns the left and right lanes with the most consistent slopes and minimal distances between them.
- **`add_to_buffer`**: Adds newly detected lines to a buffer for smoothing over time. This helps maintain consistency by averaging detected lines over multiple frames.
- **`average_lines`**: Averages the lines in the buffer to create a smooth, stable line that reduces frame-to-frame noise. This is essential in real-time applications, where lane line positions can fluctuate.
- **`draw_lines`**: Draws the detected left and right lane lines on the input image, using the intersection point of the lanes to extend the lines accurately. The lines are drawn in red to clearly visualize lane boundaries.
- **`make_line_coordinates`**: Generates line coordinates for the detected lane lines. It adjusts the line to extend from the bottom of the image to a point near the intersection of the left and right lanes, ensuring the lines fit within the image and follow the correct slope.
- **`line_intersection`**: Calculates the intersection point of two lines. It checks if the lines are parallel (no intersection) and if not, computes the point where the two lines cross. This is useful for detecting where lane lines converge (e.g., vanishing point on the horizon).
- **`line_point_distance`**: Calculates the shortest distance between a line and a point using the perpendicular distance formula. This is used to determine how close a line is to a given point, such as the intersection found in `RANSAC`.
- **`line_distance`**: Computes the distance between two approximately parallel lines by calculating the perpendicular distance between them. This is useful when comparing two lane lines to ensure they are correctly spaced apart.
- **`main`**:
  - Initializes the video feed and processes each frame using the lane detection pipeline.
  - Enhances the image to detect dashed lines as solid, continuous lines for improved visualization.
  - Outputs the processed video with lane markings.

## Demonstrating Video

- **Sample videos in Google Drive:**
  [Google Drive Link](https://drive.google.com/drive/folders/1MfBh7xroQ3lGHEUabCWor48rcw4rfypk?usp=drive_link)

  - `lane_test1.mp4` → `out_lane_test1.avi` & `Sample for test1.mov`
  - `lane_test2.mp4` → `out_lane_test2.avi` & `Sample for test2.mov`
  - `lane_test3.mp4` → `out_lane_test3.avi` & `Sample for test3.mov`

- **Optional recorded video:**
  
  **Note:** Since testing was conducted by a single person, we recorded the road footage using a WebCam and DashCam rather than capturing real-time video.

  - `real-road-1.mp4` → `out_real-road-1.avi`
  - `real-road-2.mp4` → `out_real-road-2.avi`
  - `real-road-3.mp4` → `out_real-road-3.avi`
