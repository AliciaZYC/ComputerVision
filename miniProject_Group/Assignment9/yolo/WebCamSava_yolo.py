# USAGE: python3 WebCamSave_yolo.py -f test.mp4 -o out_video.avi

import cv2
import numpy as np
import argparse

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

# Define the codec and create VideoWriter objects for each target class (only if needed)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_cat = None
out_dog = None
recording_cat = False
recording_dog = False

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Update paths if necessary

# Load COCO names (categories)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Specify the classes to detect (e.g., "cat" and "dog")
target_classes = {"cat": None, "dog": None}
target_class_ids = {classes.index(name) for name in target_classes.keys()}

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

while True:
    # Grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Flags to check if "cat" or "dog" is detected in this frame
    detected_cat = False
    detected_dog = False

    # Process detections and draw labels on frame if cat or dog is detected
    for out_layer in outs:
        for detection in out_layer:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in target_class_ids:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Determine the detected class and draw bounding box and label
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Set flags if cat or dog is detected
                if classes[class_id] == "cat":
                    detected_cat = True
                    if not recording_cat:
                        filename = f"{args.out}_cat.avi"
                        out_cat = cv2.VideoWriter(filename, fourcc, 20.0, (width, height), True)
                        recording_cat = True
                elif classes[class_id] == "dog":
                    detected_dog = True
                    if not recording_dog:
                        filename = f"{args.out}_dog.avi"
                        out_dog = cv2.VideoWriter(filename, fourcc, 20.0, (width, height), True)
                        recording_dog = True

    # Write the frame to the output video files if recording
    if recording_cat and out_cat is not None:
        out_cat.write(frame)
    if recording_dog and out_dog is not None:
        out_dog.write(frame)

# Release resources
vs.release()
if recording_cat and out_cat is not None:
    out_cat.release()
if recording_dog and out_dog is not None:
    out_dog.release()

print("Detection and recording completed.")
