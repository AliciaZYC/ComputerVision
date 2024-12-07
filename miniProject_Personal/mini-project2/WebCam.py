# USAGE: python CamTest.py

# import the necessary packages
import cv2
import time
import os
import imutils

# Open Video Camera
vs = cv2.VideoCapture(0)  # 0 is the default camera
time.sleep(2.0)

cropping = False
start_point = None
end_point = None

# loop over the frames from the video stream
while True:
    # grab the frame from video stream
    ret, frame = vs.read()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    elif key == ord("c") or key == ord("C"):
        print("Crop")
        h, w = frame.shape[:2]
        cropped = frame[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        cv2.imshow("Cropped Frame", cropped)

    elif key == ord("r") or key == ord("R"):
        print("Resize")
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (300, 300*h//w))
        cv2.imshow("Resized Frame", resized)

    elif key == ord("b") or key == ord("B"):
        print("Blur")
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        cv2.imshow("Blurred Frame", blurred)

    elif key == ord("a") or key == ord("A"):
        print("Add a box")
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (int(w * 0.1), int(h * 0.1)), (int(w * 0.9), int(h * 0.9)), (0, 255, 0), 3)
        cv2.imshow("Box Frame", frame)

    # Add text
    elif key == ord("t") or key == ord("T"):
        print("Add text")
        cv2.putText(frame, "Yichi Zhang", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.imshow("Text Frame", frame)

    elif key == ord("g") or key == ord("G"):
        print("Thresholding")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("Threshold Frame", thresh)

    # Add New Function: Detect and Draw Contours(what we've learnt)
    elif key == ord("n") or key == ord("N"):
        print("New function")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        output = frame.copy()
        for c in cnts:
            cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
        cv2.imshow("Contours", output)

# Release the video capture and close windows
vs.release()
cv2.destroyAllWindows()
