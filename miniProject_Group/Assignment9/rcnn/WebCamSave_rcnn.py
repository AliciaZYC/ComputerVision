# USAGE: python3 WebCamSave_rcnn.py -f test.mp4 -o out_video.avi

import cv2
import numpy as np
import argparse
import os
import tensorflow as tf
import time

# Non-maximum Suppression function
def non_max_suppression(boxes, overlapThresh):
    """Perform non-maximum suppression on bounding boxes."""
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name prefix")
args = parser.parse_args()

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera


# Get the default resolutions
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object if output is specified
out_filename = args.out
if out_filename:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)
else:
    out = None

# Load the pre-trained model
final_model = tf.keras.models.load_model('winebottle_light.h5')
# Set up Selective Search for region proposals
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# FPS calculation
prev_time = time.time()
while True:
    # Capture frame-by-frame
    ret, frame = vs.read()
    if not ret:
        print("Failed to capture image")
        break

    resized_frame = cv2.resize(frame, (320, 320))  # Resize for speed

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Set up Selective Search on the current frame
    ss.setBaseImage(resized_frame)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()

    # Store bounding boxes and scores
    boxes = []

    for e, result in enumerate(ssresults):
        if e < 30:  # Limit to the top 50 region proposals to maintain performance
            x, y, w, h = result
            region = resized_frame[y:y + h, x:x + w]
            resized = cv2.resize(region, (224, 224), interpolation=cv2.INTER_AREA)
            resized = np.expand_dims(resized, axis=0)

            # Model prediction
            out_model = final_model.predict(resized)
            score = out_model[0][1]  # Assuming the second value is the confidence for the object class
            if score > 0.8:  # Confidence threshold
                boxes.append([x, y, x + w, y + h, score])

    # Convert list of boxes to numpy array for NMS
    boxes = np.array(boxes)

    # Apply Non-maximum Suppression (NMS)
    nms_boxes = non_max_suppression(boxes, overlapThresh=0.2)

    scale_x = width / 320
    scale_y = height / 320
    # Draw bounding boxes on the frame
    for box in nms_boxes:
        x1, y1, x2, y2 = box[:4]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Write the frame to the output video file if specified
    if out is not None:
        out.write(frame)
    # Display the resulting frame
    cv2.imshow('WebCam Object Detection', frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
vs.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
