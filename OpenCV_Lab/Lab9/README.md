# Lab 9: Contours in OpenCV

## Overview

This lab demonstrates how to detect and draw contours in images using OpenCV. Contours are useful in object segmentation, shape analysis, and various computer vision applications. In this lab, we will learn how to retrieve, approximate, and draw contours on an image.

### Additional Information

Contours are a fundamental tool in image processing, enabling the detection of object boundaries and the extraction of shape descriptors.

## 1. Introduction to Contours

Contours are closed curves that trace the boundary of objects within an image. They capture the shape and outline of objects, which is beneficial for:

- Edge detection
- Object detection
- Object segmentation
- Object recognition
- Shape analysis
- Motion tracking

### Steps for Detecting Contours

1. **Convert the image to grayscale** and then apply **thresholding** or **edge detection** to create a binary image. This step simplifies the distinction between the object and the background.
2. **Retrieve and approximate contours** using various modes and approximation methods.
3. **Draw the contours** on the original or a blank image.

*Note*: In cases where the image has well-defined edges and minimal noise, binary conversion may be skipped.

### Additional Steps

- **Preprocessing**: Applying filters (e.g., Gaussian blur) to reduce noise before thresholding.
- **Morphological Operations**: Using dilation or erosion to close gaps or remove small artifacts.

## 2. Contour Retrieval & Approximation

### Retrieval Modes

Different modes for contour retrieval determine which contours are extracted:

- **RETR_EXTERNAL**: Retrieves only the outermost contours, ignoring any nested contours. This is useful for detecting the overall shape.
- **RETR_LIST**: Retrieves all contours without establishing any hierarchical relationship, returning a flat list of contours.
- **RETR_CCOMP**: Retrieves all contours and organizes them into a two-level hierarchy. External contours of the object (boundary) are placed in one level, and internal contours (holes) in the other level.
- **RETR_TREE**: Retrieves all contours and organizes them in a full hierarchical structure, showing relationships between nested contours.

### Hierarchy Structure

- The hierarchy is a NumPy array where each contour has information about its hierarchy relationship:
    
    ```python
    hierarchy = [[Next, Previous, First_Child, Parent], ...]
    ```
    
    - **Next**: Index of the next contour at the same hierarchical level.
    - **Previous**: Index of the previous contour at the same hierarchical level.
    - **First_Child**: Index of the first child contour.
    - **Parent**: Index of the parent contour.

### Approximation Modes

Different modes for contour approximation determine the level of detail retained:

- **CHAIN_APPROX_NONE**: Stores all contour points, providing high precision but using more memory.
- **CHAIN_APPROX_SIMPLE**: Reduces the number of points by retaining only the endpoints of lines, saving memory while still capturing the shape.
- **CHAIN_APPROX_TC89_L1** and **CHAIN_APPROX_TC89_KCOS**: Apply the Teh-Chin chain approximation algorithm to further compress the contour.

### Function Syntax

```python
contours, hierarchy = cv2.findContours(image, retrieval_mode, approximation_mode)
```

- **image**: Binary source image, typically created from thresholding or edge detection.
- **retrieval_mode**: Specifies which contours to retrieve (e.g., `cv2.RETR_EXTERNAL`).
- **approximation_mode**: Specifies the approximation method (e.g., `cv2.CHAIN_APPROX_SIMPLE`).

### Example Code

```python
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
```

## 3. Drawing Contours

Once contours are detected, they can be drawn on an image.

### Function Syntax

```python
cv2.drawContours(image, contours, contourIdx, color, thickness)
```

- **image**: The image where contours will be drawn.
- **contours**: List of contours obtained from `findContours`.
- **contourIdx**: Index of the contour to draw. Use `1` to draw all contours.
- **color**: Color of the contour lines (e.g., `(0, 255, 0)` for green).
- **thickness**: Thickness of the contour lines.

### Example Code

```python
# Draw all contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Draw a specific contour
cv2.drawContours(image, contours, contour_index, (255, 0, 0), 2)
```

## 4. Contour Analysis

Beyond detecting and drawing contours, we can analyze contours to extract useful information.

### Moments

- **Moments**: Calculate spatial moments to find properties like area, centroid, orientation, etc.

```python
M = cv2.moments(contour)
area = M['m00']
if area != 0:
    cx = int(M['m10']/M['m00'])  # Centroid x coordinate
    cy = int(M['m01']/M['m00'])  # Centroid y coordinate
else:
    cx, cy = 0, 0  # Handle division by zer
```

### Contour Properties

- **Area**: `cv2.contourArea(contour)`
- **Perimeter (Arc Length)**: `cv2.arcLength(contour, closed=True)`
- **Approximate Contour**: Simplify the contour shape.

```python
epsilon = 0.01 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
```

- **Convex Hull**: Find the convex hull of the contour.

```python
hull = cv2.convexHull(contour)
```

### Shape Matching

- **Contour Matching**: Compare shapes using methods like `cv2.matchShapes`.

```python
ret = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
```

### Bounding Shapes

- **Bounding Rectangle**: Draw a rectangle around the contour.

```python
x, y, w, h = cv2.boundingRect(contour)
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

- **Rotated Rectangle**: Minimum area rectangle that can be rotated.

```python
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
```

- **Minimum Enclosing Circle**:

```python
(x, y), radius = cv2.minEnclosingCircle(contour)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(image, center, radius, (0, 255, 0), 2)
```

## 5. Practical Applications

- **Object Detection**: Use contours to detect and isolate objects in an image.
- **Shape Analysis**: Analyze the shape of objects for classification or recognition.
- **Gesture Recognition**: Detect hand contours for recognizing gestures.
- **Motion Tracking**: Track moving objects by analyzing contours in video frames.

## 6. Tips and Best Practices

- **Preprocessing**: Proper preprocessing is crucial. Use filters to reduce noise and thresholding to get a clean binary image.
- **Hierarchy Usage**: Use contour hierarchy to differentiate between parent and child contours, especially in complex images.
- **Contour Approximation**: Simplify contours for faster processing and to focus on overall shape rather than small details.

## 7. Summary

- **Contours**: Useful for isolating and analyzing object shapes within an image.
- **Retrieval Modes**: Different modes (`RETR_EXTERNAL`, `RETR_TREE`, etc.) control whether only outer contours or hierarchical structures are retrieved.
- **Approximation Modes**: `CHAIN_APPROX_SIMPLE` reduces memory usage by simplifying contours, while `CHAIN_APPROX_NONE` retains all points for maximum precision.
- **Applications**: Contours are widely used in object detection, shape analysis, and segmentation tasks in computer vision.