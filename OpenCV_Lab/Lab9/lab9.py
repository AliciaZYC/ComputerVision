# Lab 9: Contours
import cv2

# Step 1: Load the image
# The input image is loaded in BGR format by default in OpenCV.
image = cv2.imread('input_image.jpg')

# Step 2: Convert the image to grayscale
# Converting the image to grayscale simplifies the process and is a common step before contour detection.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply binary thresholding
# Thresholding simplifies the distinction between object and background by converting the grayscale image to binary.
# Here, pixels with a value greater than 127 are set to 255 (white), and others are set to 0 (black).
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Step 4: Find contours
# Contour detection is performed on the binary image using the cv2.findContours function.
# - binary: the source binary image.
# - cv2.RETR_EXTERNAL: retrieves only the outermost contours.
# - cv2.CHAIN_APPROX_SIMPLE: removes redundant points and compresses the contour, saving memory.
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Draw contours on the original image
# The cv2.drawContours function is used to draw contours on the original image.
# - image: the original image where contours will be drawn.
# - contours: the list of contours found.
# - -1: indicates that all contours should be drawn.
# - (0, 255, 0): color of the contour (green in this case).
# - 2: thickness of the contour lines.
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Step 6: Display the result
# Display the image with contours drawn. Press any key to close the image window.
cv2.imshow("Contours", image)
cv2.imwrite("contours.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Explanation of Parameters and Modes

# cv2.findContours(image, retrieval_mode, approximation_mode)
# - image: Binary source image (created from thresholding or edge detection).
# - retrieval_mode: Determines which contours to retrieve.
#   - RETR_EXTERNAL: Retrieves only the outermost contours, ignoring nested ones.
#   - RETR_LIST: Retrieves all contours without establishing a hierarchy.
#   - RETR_TREE: Retrieves all contours and creates a hierarchical structure (useful for nested contours).
# - approximation_mode: Controls how much contour detail is retained.
#   - CHAIN_APPROX_NONE: Stores all contour points, providing high precision but consuming more memory.
#   - CHAIN_APPROX_SIMPLE: Reduces the number of points by only keeping the endpoints of lines.

# cv2.drawContours(image, contours, contourIdx, color, thickness)
# - image: The image on which to draw the contours.
# - contours: List of detected contours from cv2.findContours.
# - contourIdx: Index of the contour to draw. Use -1 to draw all contours.
# - color: Color of the contour lines (e.g., (0, 255, 0) for green).
# - thickness: Thickness of the contour lines. Thicker lines are more visible.

