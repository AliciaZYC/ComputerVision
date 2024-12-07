import cv2
import numpy as np

# Sobel Edge Detection Example
# Load an image in grayscale
image = cv2.imread('input_image.jpg', 0)

# Apply Sobel filter to detect horizontal edges
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # dx=1, dy=0 means horizontal

# Apply Sobel filter to detect vertical edges
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # dx=0, dy=1 means vertical

# Combine Sobel X and Y to get the overall edge magnitude
sobel_combined = cv2.sqrt(sobel_x**2 + sobel_y**2)

# Display Sobel result
cv2.imshow("Sobel Edge Detection", sobel_combined)
cv2.imwrite("sobelEdge.jpg",sobel_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Explanation:
# - cv2.Sobel(src, ddepth, dx, dy, ksize=3):
#   - src: Source image
#   - ddepth: Desired depth of the destination image
#   - dx: Order of the derivative in x-direction
#   - dy: Order of the derivative in y-direction
#   - ksize: Kernel size (3x3)
# - sobel_combined: Combines horizontal and vertical edges to get edge magnitude

# Canny Edge Detection Example
# Load an image in grayscale
image = cv2.imread('input_image.jpg', 0)

# Apply Canny edge detection with thresholds for edge gradient
edges = cv2.Canny(image, 100, 200)

# Display Canny result
cv2.imshow("Canny Edge Detection", edges)
cv2.imwrite("cannyEdge.jpg",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Explanation:
# - cv2.Canny(image, threshold1, threshold2):
#   - image: Source image
#   - threshold1: Lower threshold for weak edges
#   - threshold2: Upper threshold for strong edges

