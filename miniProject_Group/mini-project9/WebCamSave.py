# USAGE: python3 WebCamSave.py -f test.mp4 -o out_video.avi

# import the necessary packages
import cv2
import numpy as np
import time
import os
import argparse
import torch  # 新增，用于加载 YOLOv5 模型

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")

args = parser.parse_args()

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/Users/ziyang/Downloads/best.pt')  # 替换为 best.pt 的实际路径
model.conf = 0.4  # 设置置信度阈值，可根据需要调整

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

time.sleep(2.0)

# Get the default resolutions
width = int(vs.get(3))
height = int(vs.get(4))

# Define the codec and create a VideoWriter object
out_filename = args.out
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

# Loop over the frames from the video stream
while True:
    # Grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

    # YOLOv5 model detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Get detection results

    # Draw detection results on the image
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} ({conf * 100:.1f}%)"

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video file
    out.write(frame)

# Release the video capture object
vs.release()
out.release()
print(f"Output video saved as {out_filename}")
