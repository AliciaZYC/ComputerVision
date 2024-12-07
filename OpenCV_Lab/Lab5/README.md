# OpenCV Workshop Lab 5: Histogram

## Overview

This lab covers **histograms** in image processing, explaining how they represent the distribution of pixel intensity values. It includes the implementation of both **grayscale and color histograms**.

- **Histogram**: A graphical representation showing the frequency of pixel intensities in an image.
- **Pixel Intensity**: Ranges from 0 (black) to 255 (white).
- **Grayscale Histogram**: Shows the distribution of intensities in a grayscale image.
- **Color Histogram**: Displays the distribution of intensities across the **BGR (Blue, Green, Red)** color channels.

## Code Overview

### 1. `load_image(file_path, grayscale=False)`

This function loads an image in either **color** or **grayscale** mode.

```python
img = cv2.imread(file_path, 0)  # Grayscale
img = cv2.imread(file_path)     # Color
```

- **Returns**: The loaded image or an error message if loading fails.

---

### 2. `grayscale_histogram(img)`

This function computes and displays the histogram of a grayscale image.

```python
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()
```

- **`cv2.calcHist()`**: Computes the histogram.
- **`plt.plot()`**: Plots the histogram with the x-axis representing pixel intensity values and the y-axis showing the frequency.

---

### 3. `color_histogram(img)`

This function computes and displays histograms for each **BGR channel** of a color image.

```python
for i, col in enumerate(('b', 'g', 'r')):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
```

- **`cv2.calcHist()`**: Computes histograms for each channel.
- **Color Channels**:
  - **Blue** plotted in blue.
  - **Green** plotted in green.
  - **Red** plotted in red.
