# Lab 10: Camera

This project demonstrates how to use optical flow methods for motion detection and feature tracking using OpenCV in Python. It implements both sparse and dense optical flow techniques.

## Requirements

- Python 3.x
- OpenCV
- NumPy

Install requirements:

```bash
pip install opencv-python numpy
```

## Code Overview

### Camera Setup

The code accesses the camera using:

- `cv2.VideoCapture(0)`: Captures video from the default camera (0), or an external camera (1).

### Parameters for Optical Flow

1. **Shi-Tomasi Corner Detection** (`cv2.goodFeaturesToTrack`)
   - **`maxCorners`**: Maximum number of corners to return.
   - **`qualityLevel`**: Minimum accepted quality of corners.
   - **`minDistance`**: Minimum distance between corners.
2. **Lucas-Kanade Sparse Optical Flow** (`cv2.calcOpticalFlowPyrLK`)
   - **`prev_frame`**: Previous grayscale frame for initial points.
   - **`next_frame`**: Next grayscale frame where points are tracked.
   - **`prev_points`**: Points in `prev_frame` to track.
   - **`params`**: Window size, pyramid level, and termination criteria.
3. **Farneback Dense Optical Flow** (`cv2.calcOpticalFlowFarneback`)
   - **`prev_frame`** and **`next_frame`**: Grayscale frames.
   - **`pyr_scale`**: Scale for pyramid levels (0-1).
   - **`levels`**: Number of pyramid levels.
   - **`winsize`**: Averaging window size.
   - **`iterations`**: Refinement iterations per pyramid level.
   - **`poly_n`**: Neighborhood size for polynomial expansion.
   - **`poly_sigma`**: Standard deviation for Gaussian weighting.
   - **`flags`**: Algorithm flags.

### Usage

1. Two windows display(lab10.py):
   - **Lucas-Kanade Optical Flow (Sparse)**: Shows tracked feature points.
   - **Dense Optical Flow**: Displays flow direction and speed as a color map.
2. One window with combined result(lab.py)
3. Press `q` to exit.
