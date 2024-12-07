import cv2
import numpy as np

# Set up camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Parameters for Shi-Tomasi Corner Detection for Lucas-Kanade Optical Flow
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial points to track using Shi-Tomasi Corner Detection
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask for drawing Lucas-Kanade optical flow tracks
mask = np.zeros_like(old_frame)

while True:
    # Capture a new frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- Lucas-Kanade Optical Flow (Sparse) ----
    # Calculate optical flow using Lucas-Kanade for sparse feature points
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points (those successfully tracked)
    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks for Lucas-Kanade Optical Flow
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            dense_output = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    # ---- Dense Optical Flow ----
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to HSV format for visualization
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    # Convert flow to polar coordinates to get the magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue represents direction
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude

    # Convert HSV to BGR for visualization
    dense_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Overlay Lucas-Kanade tracks on the dense optical flow output
    combined_output = cv2.add(dense_output, mask)

    # Display the combined results
    cv2.imshow("Combined Optical Flow", combined_output)

    # Update the previous frame and points for the next loop iteration
    old_gray = frame_gray.copy()
    if good_new is not None:
        p0 = good_new.reshape(-1, 1, 2)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
