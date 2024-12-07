import cv2
import numpy as np

# Load the image
img = cv2.imread("image.jpg")

# Get image dimensions and center point
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

# Rotate the image by 45 degrees
angle = 45
scale = 1.0
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h))

# Draw a filled circle at the center
cv2.circle(rotated_img, center, 50, (0, 255, 0), -1)  # Green circle

# Draw a full ellipse
cv2.ellipse(rotated_img, (w // 2, h // 2 + 100), (80, 40), 
            0, 0, 360, (255, 0, 0), 2)  # Blue ellipse

# Draw a half ellipse
cv2.ellipse(rotated_img, (w // 2, h // 2 + 200), (100, 50), 
            0, 0, 180, (0, 0, 255), 2)  # Red half-ellipse

# Display the final image
cv2.imshow("Rotated Image with Shapes", rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("lab_3_practice_image.jpg", rotated_img)