import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(image, title):
    """Display an image using matplotlib."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load an image
image = cv2.imread("sample_image.jpg")

# 1. Apply Averaging (Box Filter)
blurred_avg = cv2.blur(image, (5, 5))
display_image(cv2.cvtColor(blurred_avg, cv2.COLOR_BGR2RGB), "Averaging (Box Filter)")

# 2. Apply Gaussian Blurring
blurred_gauss = cv2.GaussianBlur(image, (5, 5), 0)
display_image(cv2.cvtColor(blurred_gauss, cv2.COLOR_BGR2RGB), "Gaussian Blurring")

# 3. Apply Median Blurring
blurred_median = cv2.medianBlur(image, 5)
display_image(cv2.cvtColor(blurred_median, cv2.COLOR_BGR2RGB), "Median Blurring")

# 4. Apply Bilateral Filtering
blurred_bilateral = cv2.bilateralFilter(image, 9, 75, 75)
display_image(cv2.cvtColor(blurred_bilateral, cv2.COLOR_BGR2RGB), "Bilateral Filtering")