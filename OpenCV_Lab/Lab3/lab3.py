import cv2
import numpy as np

# Load an image
img = cv2.imread("image.jpg")

# Resize the image
resized_img = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)

# Crop a region from the image
cropped_img = img[50:200, 100:300]

# Rotate the image 90 degrees clockwise
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Draw a line and a rectangle
cv2.line(img, (0, 0), (300, 300), (0, 255, 0), 3)
cv2.rectangle(img, (50, 50), (250, 150), (255, 0, 0), 2)

# Add text to the image
cv2.putText(img, "Hello, OpenCV!", (50, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display all modified images
cv2.imshow("Resized", resized_img)
cv2.imshow("Cropped", cropped_img)
cv2.imshow("Rotated", rotated_img)
cv2.imshow("Gray", gray_img)
cv2.imshow("Final Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("lab_3_image.jpg", img)