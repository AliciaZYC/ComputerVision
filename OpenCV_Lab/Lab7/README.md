# OpenCV Workshop Lab 7: Thresholding

## Overview

This lab introduces **thresholding techniques** used in image processing for **segmentation**. These techniques convert grayscale images into binary images by comparing pixel values to a threshold. The lab covers:

- **Simple Thresholding**: Uses a fixed threshold value for the entire image.
- **Adaptive Thresholding**: Adjusts the threshold for different regions of the image.
- **Otsu’s Binarization**: Automatically calculates the optimal threshold value.

### Additional Information

Thresholding is a fundamental technique in image processing and computer vision. It's often used as a pre-processing step for further analysis, such as feature extraction or object detection.

### Applications of Thresholding

- **Document Scanning**: Binarization of scanned documents to enhance text readability.
- **Medical Imaging**: Segmenting regions of interest, such as separating tissues in MRI scans.
- **Object Detection**: Isolating objects from the background in preparation for contour detection or object recognition.
- **Image Preprocessing**: Simplifying images to reduce computational complexity in subsequent processing steps.

## Code Overview

### 1. `simple_threshold(image)`

This function applies **simple thresholding** to the input image.

- **Binary Threshold**: Sets pixel values greater than the threshold to 255 and others to 0.
- **Inverse Binary Threshold**: The opposite of binary thresholding.
- **Truncation Threshold**: Sets pixel values greater than the threshold to the threshold value.
- **To Zero Threshold**: Sets pixel values less than the threshold to zero.
- **To Zero Inverse Threshold**: Sets pixel values greater than the threshold to zero.

**Example code:**

```python
_, thresh_binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
_, thresh_binary_inv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
_, thresh_trunc = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
_, thresh_tozero = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
_, thresh_tozero_inv = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
```

### Parameters

- **`src`**: Source grayscale image.
- **`thresh`**: Threshold value.
- **`maxval`**: Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
- **`type`**: Thresholding type (e.g., `cv2.THRESH_BINARY`).

### Use Cases

- **Simple Thresholding** is effective when the image has uniform lighting conditions and the foreground and background are well-separated in intensity.

### 2. `adaptive_threshold(image)`

This function applies **adaptive thresholding**, helpful when lighting conditions vary across the image. It uses two methods:

- **Mean Adaptive Thresholding**: Computes the mean of the neighborhood values.
- **Gaussian Adaptive Thresholding**: Computes a weighted sum of neighborhood values.

```python
thresh_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, blockSize=11, C=2)
thresh_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, blockSize=11, C=2)
```

### Parameters

- **`maxValue`**: Non-zero value assigned to the pixels for the `THRESH_BINARY` and `THRESH_BINARY_INV` thresholding types.
- **`adaptiveMethod`**: Adaptive thresholding algorithm to use (`cv2.ADAPTIVE_THRESH_MEAN_C` or `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`).
- **`thresholdType`**: Type of thresholding to apply (`cv2.THRESH_BINARY` or `cv2.THRESH_BINARY_INV`).
- **`blockSize`**: Size of a pixel neighborhood used to calculate a threshold value (must be an odd number >= 3).
- **`C`**: Constant subtracted from the mean or weighted sum.

### Choosing Parameters

- **Block Size (`blockSize`)**: Determines the size of the neighborhood area. A larger block size may smooth out local variations, whereas a smaller block size may capture fine details but can be sensitive to noise.
- **Constant (`C`)**: Adjusts the threshold value. A positive value of `C` decreases the threshold, making more pixels white. A negative value of `C` increases the threshold, making more pixels black.

### Use Cases

- **Adaptive Thresholding** is suitable for images with varying illumination, such as images with shadows or uneven lighting.

### 3. `otsu_threshold(image)`

This function applies **Otsu’s Binarization** to determine the best threshold value based on the image’s histogram.

- **Otsu's Method**: Assumes that the image contains two classes of pixels and calculates the optimal threshold separating these two classes so that their combined intra-class variance is minimal.

**Example code:**

```python
_, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Notes

- When using Otsu's method, the threshold value provided is ignored. The algorithm calculates the optimal threshold value automatically.
- Otsu's method is most effective when the histogram of the image has a bimodal distribution.

### Combining Otsu's Method with Gaussian Blur

To improve the result of Otsu's thresholding, especially in noisy images, it's common to apply a Gaussian blur before thresholding.

```python
blur = cv2.GaussianBlur(image, (5, 5), 0)
_, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Comparison of Thresholding Techniques

| Method                | Advantages                          | Disadvantages                                 |
| --------------------- | ----------------------------------- | --------------------------------------------- |
| Simple Thresholding   | Fast and straightforward            | Not effective with varying illumination       |
| Adaptive Thresholding | Handles varying lighting conditions | Computationally more intensive                |
| Otsu’s Binarization   | Automatic threshold selection       | Assumes bimodal histogram; sensitive to noise |

### Practical Tips

- Always visualize the histogram of the image to understand its intensity distribution.
- Preprocessing steps like noise reduction (e.g., Gaussian blur) can improve thresholding results.
- Experiment with different parameters (block size, `C`, etc.) to achieve the best result for your specific application.

## Summary

Thresholding is a powerful tool in image segmentation, enabling the separation of objects from the background. By understanding and choosing the appropriate thresholding method and parameters, one can significantly improve the performance of subsequent image processing tasks.
