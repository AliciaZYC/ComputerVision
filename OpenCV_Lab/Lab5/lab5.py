import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(file_path, grayscale=False):
    """Load an image in color or grayscale mode."""
    if grayscale:
        img = cv2.imread(file_path, 0)
    else:
        img = cv2.imread(file_path)
    if img is None:
        print(f"Error loading image: {file_path}")
    return img

def grayscale_histogram(img):
    """Compute and display the grayscale histogram."""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def color_histogram(img):
    """Compute and display the color histogram."""
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
    plt.title('Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def main():
    # Load images in grayscale and color mode
    grayscale_img = load_image("image.jpg", grayscale=True)
    color_img = load_image("image.jpg")

    if grayscale_img is not None:
        # Display the grayscale histogram
        grayscale_histogram(grayscale_img)

    if color_img is not None:
        # Display the color histogram
        color_histogram(color_img)

if __name__ == "__main__":
    main()
