import cv2
import sys
import imutils

img = cv2.imread("images/cake.jpg")

if img is None:
    sys.exit("Could not read the image.")

(h, w, d) = img.shape
# print("width=", w, "height=", h, "depth=", d)

# cv2.imshow("Display window", img)
# k = cv2.waitKey(0)

# if k == ord("s"):
#     cv2.imwrite("flower-copy.png", img)

# 1) Show Region of Interest (ROI):
roi = img[430:1230, 300:1100]
# cv2.imshow("ROI", roi)
# cv2.waitKey(0)

# 2) Resize Image:
(h1, w1, d1) = roi.shape
# print("parameter of ROI: width=", w1, "height=", h1, "depth=", d1)
# parameter of ROI: width= 800 height= 800 depth= 3
# dim = 200 = 800 * 200 / 800, keep aspect ratio
resized = cv2.resize(roi, (200, 200))
# cv2.imshow("Resized Image", resized)
# cv2.waitKey(0)

# 3) Rotate Image: 
# w = 200, h = 200
# center = (w // 2, h // 2) = (100, 100)
center = (100, 100)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(resized, M, (200, 200))
# cv2.imshow("Rotated Image", rotated)
# cv2.waitKey(0)

# 4) Smooth Image:
blurred = cv2.GaussianBlur(img, (11, 11), 0)
# cv2.imshow("Blurred", blurred)
# cv2.waitKey(0)

# 5) Drawing:
output = img.copy()
cv2.rectangle(output, (300, 430), (1100, 1230),(255, 0, 0), 4)
cv2.circle(output, (500, 500), 100, (0, 255, 0), -1)
cv2.line(output, (50, 50), (1200, 1200), (0, 0, 255), 5)
# cv2.imshow("Drawing", output)
# cv2.waitKey(0)

# 6) Add Text:
text_added = img.copy()
cv2.putText(text_added, "Yichi Zhang", (100, 100),
cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 4)
# cv2.imshow("Text", text_added)
# cv2.waitKey(0)

# 7) Convert to Grayscale:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray", gray)
# cv2.waitKey(0)

# 8) Edge Detection:
edged = cv2.Canny(gray, 250, 500)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)

# 9) Thresholding:
thresh = cv2.threshold(gray, 125, 255,cv2.THRESH_BINARY)[1]
# cv2.imshow("Thresh", thresh)
# cv2.waitKey(0)

# 10) Detect and Draw Contours: 
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = img.copy()
for c in cnts:
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
cv2.imshow("Contours", output)
cv2.waitKey(0)
