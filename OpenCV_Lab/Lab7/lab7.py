import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image in grayscale
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

def simple_threshold(image):
    """Applies simple binary thresholding."""
    _, thresh_simple = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh_simple, cmap='gray')
    plt.title('Simple Thresholding')
    plt.show()

def adaptive_threshold(image):
    """Applies adaptive mean and Gaussian thresholding."""
    thresh_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    thresh_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    plt.subplot(1, 2, 1)
    plt.imshow(thresh_mean, cmap='gray')
    plt.title('Adaptive Mean Thresholding')

    plt.subplot(1, 2, 2)
    plt.imshow(thresh_gaussian, cmap='gray')
    plt.title('Adaptive Gaussian Thresholding')

    plt.show()

def otsu_threshold(image):
    """Applies Otsu’s binarization method."""
    _, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(thresh_otsu, cmap='gray')
    plt.title('Otsu’s Binarization')
    plt.show()

# Call the functions to demonstrate different thresholding methods
simple_threshold(image)
adaptive_threshold(image)
otsu_threshold(image)
