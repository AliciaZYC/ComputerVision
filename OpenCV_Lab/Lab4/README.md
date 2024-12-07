# OpenCV Workshop Lab 4: Image Processing

## Overview

This lab focuses on **feature detection and matching** techniques in OpenCV. It demonstrates the use of:

- **Harris Corner Detection** for identifying corner-like structures.
- **SIFT (Scale-Invariant Feature Transform)** to detect and describe features.
- **FLANN (Fast Library for Approximate Nearest Neighbors)** for efficient feature matching between images.

## Code Overview

### 1. `harris_corner_detection()`

This function applies Harris Corner Detection, highlighting corners in red.

```python
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
image[dst > 0.01 * dst.max()] = [0, 0, 255]
```

### 2. `sift_detection()`

This function uses SIFT to detect keypoints and draw them on the image.

```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
img_sift = cv2.drawKeypoints(image, keypoints, None)
```

### 3. `flann_match()`

Performs FLANN-based matching between two images using SIFT descriptors.

```python
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
```
