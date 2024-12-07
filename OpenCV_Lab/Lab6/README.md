# OpenCV Workshop Lab 6: Smoothing and Blurring

## Overview

This lab focuses on **smoothing and blurring techniques** using OpenCV in Python. These techniques help **reduce noise** in images, improving feature extraction, edge detection, and object recognition.

- **Averaging (Box Filter)**: Replaces each pixel by the average of its neighboring pixels. Useful for **basic noise reduction**.
- **Gaussian Blurring**: Applies a Gaussian function to the image. Preserves edges better than averaging.
- **Median Blurring**: Replaces pixel values with the **median** of the neighborhood. Effective for removing **salt-and-pepper noise**.
- **Bilateral Filtering**: Smooths the image while preserving edges. Useful for applications requiring noise reduction with **edge preservation**.

## Code Overview

### 1. `display_image()`

This function uses **matplotlib** to display images in grayscale or color.

```python
def display_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
```

### 2. Apply Averaging (Box Filter)

- Uses the `cv2.blur()` function to apply a box filter.
- Demonstrates **basic blurring**.

```python
blurred_avg = cv2.blur(image, (5, 5))
display_image(cv2.cvtColor(blurred_avg, cv2.COLOR_BGR2RGB), "Averaging (Box Filter)")
```

### 3. Apply Gaussian Blurring

- Applies a **Gaussian function** to the image for **better edge preservation**.

```python
blurred_gauss = cv2.GaussianBlur(image, (5, 5), 0)
display_image(cv2.cvtColor(blurred_gauss, cv2.COLOR_BGR2RGB), "Gaussian Blurring")
```

### 4. Apply Median Blurring

- Uses `cv2.medianBlur()` to remove **salt-and-pepper noise**.

```python
blurred_median = cv2.medianBlur(image, 5)
display_image(cv2.cvtColor(blurred_median, cv2.COLOR_BGR2RGB), "Median Blurring")
```

### 5. Apply Bilateral Filtering

- Smooths the image while keeping edges sharp.

```python
blurred_bilateral = cv2.bilateralFilter(image, 9, 75, 75)
display_image(cv2.cvtColor(blurred_bilateral, cv2.COLOR_BGR2RGB), "Bilateral Filtering")
```
