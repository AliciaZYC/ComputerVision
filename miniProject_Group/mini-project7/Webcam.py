# USAGE: python3 Webcam.py -f test_video.mp4 -o output_video.avi
# USAGE: python3 Webcam.py -f test.mp4 -o output_video1.avi

# Import the necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import time
import argparse
from PIL import Image

# Load class names from 'classes.txt'
with open('classes.txt', 'r') as f:
    classes = [line.strip() for line in f]

# Define the neural network (same as in your training script)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)    # Conv layer 1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)   # Conv layer 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Conv layer 3
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)        # FC layer 1
        self.fc2 = nn.Linear(1024, 256)                # FC layer 2
        self.fc3 = nn.Linear(256, len(classes))        # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv layer 1 and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv layer 2 and pooling
        x = self.pool(F.relu(self.conv3(x)))  # Conv layer 3 and pooling
        x = torch.flatten(x, 1)               # Flatten the tensor
        x = F.relu(self.fc1(x))               # FC layer 1
        x = F.relu(self.fc2(x))               # FC layer 2
        x = self.fc3(x)                       # Output layer
        return x

# Check if CUDA is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
net = Net()
net.load_state_dict(torch.load('best_model.pth', map_location=device))
net.to(device)
net.eval()  # Set the model to evaluation mode

# Define image transformations (without data augmentation)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")
args = parser.parse_args()

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

# Verify that the video source is opened successfully
if not vs.isOpened():
    print("Error: Could not open video source.")
    exit()

time.sleep(2.0)

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

try:
    # Loop over the frames from the video stream
    while True:
        # Grab the frame from video stream
        ret, frame = vs.read()
        if not ret:
            break

        # Convert the image from BGR (OpenCV format) to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the image to PIL Image
        img = Image.fromarray(img)
        # Apply the transformations
        img = transform(img)
        # Add batch dimension and move to device
        img = img.unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = net(img)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            # Get the highest probability and its index
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = classes[predicted.item()]
            confidence_score = confidence.item()

        # Overlay the predicted class and confidence on the frame
        text = f'{predicted_class}: {confidence_score * 100:.2f}%'
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame to the output video file if specified
        if out is not None:
            out.write(frame)

        # Show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the 'q' key was pressed, break from the loop
        if key == ord("q"):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the video capture object and close windows
    vs.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
