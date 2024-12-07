# Mini-Project 11 Follow Everything

## Team Members

- **Derek Zhang**
- **Yichi Zhang**
- **Yiling Hu**
- **Ziyang Liang**

---

## **Project Overview**

This project demonstrates a Python-based real-time object detection and tracking application using:

- **YOLOv8**: For object detection.
- **Lucas-Kanade Optical Flow**: For tracking detected objects across video frames.

The goal is to detect all objects in the frame using a pre-trained YOLOv8 model and track the movements of specific classes (e.g., 'person') using optical flow, enabling continuous monitoring of selected objects.

---

## **Features**

1. **Object Detection**:
   - Utilizes the pre-trained YOLOv8 model for detecting objects in a video stream.
   - **Detects and displays bounding boxes for all objects** in the frame with class labels and confidence scores.

2. **Selective Object Tracking**:
   - **Tracks only specified classes** (e.g., 'person') across frames using Lucas-Kanade optical flow.
   - Supports tracking of multiple objects simultaneously.

3. **Tracking Controls**:
   - **Press `t`** to start tracking.
   - **Press `t` again** to stop tracking and clear the screen.
   - **Press `c`** to clear the current tracking data and trajectories without stopping tracking.

4. **Performance Monitoring**:
   - Displays Frames Per Second (**FPS**) on the top-left corner of the video feed.

5. **Video Input**:
   - Supports real-time webcam input.

---

## **Setup Instructions**

1. **Install Required Libraries**:

   Install the dependencies listed below:

   ```bash
   pip install ultralytics opencv-python scipy numpy
   ```

   - **Note**: Ensure you have **Python 3.6** or higher.

2. **Download Pre-trained YOLOv8 Model**:

   The script uses the `yolov8n.pt` model for object detection:

   ```python
   yolo_model = YOLO('yolov8n.pt')
   ```

   - You can switch to other YOLOv8 models if desired:

     - `yolov8n.pt`: Nano (fastest, least accurate)
     - `yolov8s.pt`: Small (~20 fps)
     - `yolov8m.pt`: Medium (~15 fps)
     - `yolov8l.pt`: Large
     - `yolov8x.pt`: Extra-large (most accurate, slowest)

   - **Note**: Using larger models may reduce performance (lower FPS).

3. **Run the Script**:

   Execute the script with the following command:

   ```bash
   python3 webcam.py
   ```

   - The script uses the default webcam as input.

---

## **Usage Instructions**

1. **Start the Application**:

   - Run the script to start the video feed and object detection.

2. **Tracking Controls**:

   - **Press `t`** to **toggle tracking** on or off.
     - When tracking is enabled, the script will detect and track specified objects automatically.
     - When tracking is disabled, all tracking data and trajectories are cleared.
   - **Press `c`** to **clear the current tracking data and trajectories** without disabling tracking.

3. **Object Detection and Tracking**:

   - **Detection**:
     - The script detects **all objects** in the frame and displays bounding boxes with labels.
   - **Tracking**:
     - When tracking is enabled, the script tracks only the specified classes (e.g., 'person').
     - Tracking trajectories are displayed as green lines.
     - Bounding boxes and labels for tracked objects are highlighted differently to distinguish them from other detections.

4. **Performance Monitoring**:

   - The **FPS (Frames Per Second)** is displayed on the top-left corner of the video feed.

5. **Exit**:

   - **Press `q`** to quit the application.

---

## **Additional Information**

- **Classes to Track**:

  - By default, the script is set to track the **'person'** class.
  - You can modify the `classes_to_track` variable in the script to track other classes:

    ```python
    classes_to_track = ['person', 'car']
    ```

  - **Available Classes**: Refer to the YOLOv8 model's documentation for a list of detectable classes.

- **Visualization Details**:

  - **Detection Bounding Boxes**:
    - **Color**: Blue
    - **Description**: Drawn for all detected objects not being tracked.
  - **Tracking Bounding Boxes**:
    - **Color**: Yellow
    - **Description**: Drawn for objects being tracked (specified classes).
  - **Tracking Trajectories**:
    - **Color**: Green lines
    - **Description**: Shows the movement path of tracked objects.

- **Model Performance**:

  - The script is optimized for better performance by:

    - Using a smaller YOLOv8 model (`yolov8n.pt`).
    - Resizing frames to a lower resolution (e.g., 640x480).
    - Running YOLO detection every **3 frames** to reduce computational load.
    - Adjusting optical flow parameters for faster computation.

  - **Hardware Acceleration**:

    - For improved performance, especially when using larger models, run the script on a machine with a compatible **GPU**.
    - Ensure that PyTorch and other libraries are configured to utilize GPU acceleration.

---

## **Example Commands**

1. **Run with Webcam Input**:

   ```bash
   python3 webcam.py
   ```

---

## **Script Overview**

Below is a brief overview of the key components of the `webcam.py` script:

- **Import Statements**:

  ```python
  import cv2
  import numpy as np
  from ultralytics import YOLO
  from scipy.optimize import linear_sum_assignment
  import time
  ```

- **Initialization**:

  - Load the YOLOv8 model.
  - Initialize variables for object tracking, optical flow parameters, and FPS calculation.

- **Main Loop**:

  - Capture frames from the webcam.
  - Resize frames for performance optimization.
  - Convert frames to grayscale for optical flow computation.
  - Handle tracking based on the `tracking_enabled` state.
    - When tracking is enabled:
      - Run YOLO detection every 3 frames.
      - Perform data association and track management.
      - Apply Lucas-Kanade optical flow to track objects between detections.
      - Draw bounding boxes, labels, and trajectories.
    - When tracking is disabled:
      - Display the video feed without tracking annotations.
  - Calculate and display FPS.
  - Handle key presses for user controls (`t`, `c`, `q`).

- **Key Features**:

  - **Detect All Objects**: The script detects all objects and displays bounding boxes and labels for them.
  - **Selective Tracking**: Only specified classes are tracked, reducing computational load.
  - **Toggle Tracking**: Press `t` to start or stop tracking.
  - **Clear Tracking Data**: Press `c` to clear tracking data without stopping tracking.
  - **Exit**: Press `q` to quit the application.

---

## **Dependencies and Requirements**

- **Python Version**: Ensure you are using **Python 3.6** or higher.

- **Required Libraries**:

  - `ultralytics`: For YOLOv8 object detection.
  - `opencv-python`: For video capture and image processing.
  - `numpy`: For numerical operations.
  - `scipy`: For the linear assignment problem in data association.
  - `time`: For FPS calculation.

- **Installation Command**:

  ```bash
  pip install ultralytics opencv-python scipy numpy
  ```

---

## **Customization**

- **Adjust Detection Frequency**:

  - Modify how often YOLO detection runs by changing the frame interval:

    ```python
    if frame_count % 3 == 0:
        # Run YOLO detection
    ```

  - Increase the number for less frequent detection (improves FPS but may reduce tracking accuracy).

- **Change Frame Resolution**:

  - Adjust the `desired_width` and `desired_height` variables to change the processing resolution:

    ```python
    desired_width = 640
    desired_height = 480
    ```

  - Lower resolutions can improve performance at the cost of detection accuracy.

- **Optical Flow Parameters**:

  - Tweak `lk_params` to optimize the optical flow algorithm for your specific use case:

    ```python
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    ```

- **Adjust Confidence Threshold**:

  - Modify the confidence threshold for detections to filter out low-confidence detections:

    ```python
    if confidence > 0.5:
        # Process detection
    ```

---

## **Troubleshooting**

- **Low FPS**:

  - Use a smaller YOLO model (`yolov8n.pt`).
  - Reduce the frame resolution.
  - Ensure that your system has sufficient resources or consider using a machine with a dedicated GPU.

- **Module Not Found Errors**:

  - Verify that all required libraries are installed.
  - Use `pip list` to check installed packages and their versions.

- **Camera Access Issues**:

  - Ensure that your webcam is properly connected and not being used by another application.
  - Check permissions if running on an operating system that restricts camera access.

- **Bounding Boxes Flashing**:

  - The script addresses flashing by storing and consistently displaying the last detections.
  - If you notice flashing, ensure that the code storing and drawing `last_detections` is correctly implemented.

---

## **Video Demo**
[Google Drive](https://drive.google.com/drive/folders/10ehLSJHL3uk9QPhVTZnkJCsxqea7fR6Z?usp=share_link)
