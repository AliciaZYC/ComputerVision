
# USAGE: python3 WebCamLive.py
import cv2
import torch

model_path = '/Users/ziyang/Downloads/best1.pt'  # Path to your YOLOv5 model weights
model = torch.hub.load('ultralytics/yolov5', 'custom',path=model_path, force_reload=True)  # Load YOLOv5 small model

# Access the webcam
cap = cv2.VideoCapture(0)  # Change to the appropriate camera index if you have multiple cameras

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Run loop to get frames from the webcam and perform YOLO detection
try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform inference
        results = model(frame)

        # Display the results on the frame
        result_img = results.render()[0]  # results.render() returns list of frames

        # Display the resulting frame
        cv2.imshow('YOLO Detection', result_img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
