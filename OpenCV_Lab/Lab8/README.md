# OpenCV Workshop Lab 8: Edge Detection

## Edge Detection Overview

Edge detection aims to find areas in an image where pixel intensity changes sharply, indicating boundaries or edges. This project specifically demonstrates:

- **Sobel Edge Detection**: Efficient and directional gradient-based edge detection.
- **Canny Edge Detection**: A multi-step algorithm with noise reduction and more refined control over detected edges.

### Additional Information

Edge detection is a critical step in many image processing and computer vision applications, such as:

- **Feature Extraction**: Identifying important features for object recognition.
- **Image Segmentation**: Dividing an image into meaningful regions.
- **Image Registration**: Aligning images from different datasets.

## Techniques Implemented

### Sobel Edge Detection

The Sobel operator calculates image gradients in the horizontal (X) and vertical (Y) directions. It uses two convolution kernels to compute these gradients, which highlight intensity changes.

### Mathematical Background

- The Sobel operator uses convolution with a pair of 3x3 kernels (one for horizontal changes, one for vertical).
- The gradient magnitude is calculated as:

\[
G = \sqrt{G_x^2 + G_y^2}
\]

- The gradient direction is:

\[
\theta = \arctan\left(\frac{G_y}{G_x}\right)
\]

### Kernel Examples

- **Sobel X Kernel**:

\[
\begin{bmatrix}
-1 & 0 & +1 \\
-2 & 0 & +2 \\
-1 & 0 & +1
\end{bmatrix}
\]

- **Sobel Y Kernel**:

\[
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
+1 & +2 & +1
\end{bmatrix}
\]

### Code Implementation

```python
# Convert to grayscale if necessary
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sobel Edge Detection Example
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Detect vertical edges
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Detect horizontal edges

# Compute the gradient magnitude
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Optionally convert to 8-bit image
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
```

### Practical Tips

- **Kernel Size (`ksize`)**: Larger kernels can detect larger edges but may blur the result.
- **Image Depth (`ddepth`)**: Use `cv2.CV_64F` to avoid data overflow.

### Canny Edge Detection

The Canny Edge Detection algorithm is a more advanced method with multiple steps:

1. **Noise Reduction**: Applies Gaussian filtering to reduce noise.
2. **Gradient Calculation**: Calculates intensity gradients in the image.
3. **Non-Maximum Suppression**: Keeps only the strongest edges and removes those that are weaker.
4. **Double Thresholding**: Sets high and low thresholds for strong and weak edges.
5. **Edge Tracking by Hysteresis**: Retains edges that connect to strong edges.

### Steps in Detail

- **Noise Reduction**: Uses a Gaussian filter to smooth the image and reduce noise, preventing false edge detection.
  ```python
  blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
  ```
- **Gradient Calculation**: Similar to Sobel operator; computes gradients in the X and Y directions.
- **Non-Maximum Suppression**: Thin out the edges to have a pixel-wide edge line.
- **Double Thresholding**: Classifies pixels as strong, weak, or non-relevant based on two thresholds.
- **Edge Tracking by Hysteresis**: Weak edges connected to strong edges are considered true edges.

### Code Implementation

```python
# Convert to grayscale if necessary
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)

# Canny Edge Detection Example
edges = cv2.Canny(blurred_image, 100, 200)  # Apply Canny with thresholds 100 and 200
```

### Choosing Threshold Values

- **Lower Threshold (`threshold1`)**: Determines the minimum gradient intensity to consider as an edge.
- **Upper Threshold (`threshold2`)**: Determines the minimum gradient intensity to consider as a strong edge.
- **Ratio**: A common practice is to set the upper threshold to be 2 or 3 times the lower threshold.

### Practical Tips

- **Automatic Thresholding**: If unsure about threshold values, use methods like Otsu's method to estimate them.
  ```python
  # Automatically determine lower and upper thresholds
  median_val = np.median(blurred_image)
  lower = int(max(0, (1.0 - sigma) * median_val))
  upper = int(min(255, (1.0 + sigma) * median_val))
  edges = cv2.Canny(blurred_image, lower, upper)
  ```
- **Edge Connectivity**: Adjusting the thresholds affects the connectivity of edges; lower thresholds may result in more connected edges but can introduce noise.

## Comparison of Sobel and Canny Edge Detection

| Feature              | Sobel                     | Canny                                         |
| -------------------- | ------------------------- | --------------------------------------------- |
| Gradient Calculation | Simple intensity gradient | Multi-step filtering and gradient calculation |
| Noise Sensitivity    | Sensitive                 | Uses Gaussian to reduce noise                 |
| Edge Quality         | Thicker edges             | Thin, precise edges                           |
| Control              | Limited                   | User-defined thresholds for fine control      |

### Other Edge Detection Methods

- **Prewitt Operator**: Similar to Sobel but with different kernels.
- **Roberts Cross Operator**: Uses 2x2 kernels, sensitive to noise.
- **Laplacian of Gaussian (LoG)**: Applies Gaussian smoothing followed by Laplacian operator; captures edges with more accuracy.
- **Zero Crossing**: Detects edges by finding zero crossings in the second derivative.

### Applications

- **Medical Imaging**: Edge detection helps in identifying tumors or anatomical structures.
- **Autonomous Driving**: Detecting lane lines and road boundaries.
- **Image Compression**: Edges are essential features in preserving image quality during compression.
- **Object Recognition**: Edges are fundamental in recognizing and classifying objects.

## Summary

Edge detection is a crucial step in many computer vision tasks. The choice between Sobel and Canny depends on the specific requirements of the application. Sobel is computationally efficient for detecting basic edges, while Canny provides more accurate and thin edges suitable for complex images.
