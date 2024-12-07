# OpenCV Workshop Lab 2: Loading, Displaying, and Saving Images

## Overview

This lab demonstrates **how to load, display, and save images using OpenCV** in Python. It covers basic functionalities:

- Loading images in both color and grayscale formats.
- Displaying the images using OpenCV functions.
- Saving the images with a new filename and format.

## Code Overview

### 1. `load_images()`

This function loads two versions of the same image:

- **Color Image:** Loaded using `cv2.IMREAD_COLOR`.
- **Grayscale Image:** Loaded using `cv2.IMREAD_GRAYSCALE`.

```python
img = cv2.imread("cake.jpg", cv2.IMREAD_COLOR)
gray_img = cv2.imread("cake.jpg", cv2.IMREAD_GRAYSCALE)
```

If the image cannot be loaded, it prints an error message.

### 2. `display_images()`

This function displays the loaded images using OpenCV:

- Opens two separate windows, one for the color image and one for the grayscale image.
- Waits for a key press before closing the windows.

```python
cv2.imshow("Color Image", img)
cv2.imshow("Grayscale Image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 3. `save_images()`

This function saves the color image with a new filename:

- Uses the `cv2.imwrite()` function to save the image.
- Prints a success message upon saving.

```python
cv2.imwrite("lab_2_image.jpg", img)
```
