import cv2
import numpy as np
import time


def translation(frame, horizontal, vertical):
    """
    Shift the frame horizontally and/or vertically.

    Args:
        frame (ndarray): Input image.
        horizontal (int): Pixels to shift horizontally.
        vertical (int): Pixels to shift vertically.

    Returns:
        ndarray: Transformed image with translation.
    """
    rows, cols = frame.shape[:2]
    M = np.float32([[1, 0, horizontal], [0, 1, vertical]])  # Translation matrix
    return cv2.warpAffine(frame, M, (cols, rows))


def rotation(frame, degree):
    """
    Rotate the frame by a specified degree.

    Args:
        frame (ndarray): Input image.
        degree (float): Degrees to rotate the image.

    Returns:
        ndarray: Rotated image.
    """
    rows, cols = frame.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)  # Rotation matrix
    return cv2.warpAffine(frame, M, (cols, rows))


def scale(frame, factor):
    """
    Scale the frame by a specified factor.

    Args:
        frame (ndarray): Input image.
        factor (float): Scaling factor.

    Returns:
        ndarray: Scaled image, cropped to original size.
    """
    rows, cols = frame.shape[:2]
    scaled = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)  # Scaling
    new_rows, new_cols = scaled.shape[:2]
    start_x = max((new_cols - cols) // 2, 0)
    start_y = max((new_rows - rows) // 2, 0)
    return scaled[start_y:start_y + rows, start_x:start_x + cols]  # Cropped to original size


def perspective(frame, top_left, top_right, bottom_left, bottom_right):
    """
    Apply a perspective transformation to simulate a different viewpoint.

    Args:
        frame (ndarray): Input image.
        top_left, top_right, bottom_left, bottom_right (tuple): Coordinates for perspective transform.

    Returns:
        ndarray: Image with perspective transformation.
    """
    rows, cols = frame.shape[:2]

    # Perspective Transform: Warp to simulate a different viewpoint
    pts1 = np.float32([[50, 50], [cols - 50, 50], [50, rows - 50], [cols - 50, rows - 50]])
    pts2 = np.float32([
        [50 + top_left[0], 50 + top_left[1]],
        [cols - 50 + top_right[0], 50 + top_right[1]],
        [50 + bottom_left[0], rows - 50 + bottom_right[1]],
        [cols - 50 + bottom_right[0], rows - 50 + bottom_right[1]]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2) # Perspective matrix
    return cv2.warpPerspective(frame, M, (cols, rows))


def apply_transformation(frame, curr_mode):
    """
    Apply the selected transformation to the frame based on the current mode.

    Args:
        frame (ndarray): Input image.
        curr_mode (str): Transformation mode ('translation', 'rotation', 'scaling', 'perspective').

    Returns:
        ndarray: Transformed image.
    """
    wrap_dict = {
        'translation': lambda: translation(frame, 100, 200),
        'rotation': lambda: rotation(frame, 45),
        'scaling': lambda: scale(frame, 2),
        'perspective': lambda: perspective(frame, (80, 50), (100, 100), (30, 10), (200, 40))
    }
    return wrap_dict.get(curr_mode)() if curr_mode in wrap_dict else frame


def set_mode(curr_mode, key):
    """
    Set the transformation mode based on key input.

    Args:
        curr_mode (str): Current transformation mode.
        key (str): Key pressed to switch modes ('t', 'r', 's', 'p').

    Returns:
        str: Updated transformation mode.
    """
    mode_dict = {'t':'translation', 'r':'rotation', 's':'scaling', 'p':'perspective'}
    if key not in mode_dict:
        return curr_mode
    else:
        new_mode = mode_dict.get(key)
    return 'original' if curr_mode == new_mode else new_mode


def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Use index 0 for default camera

    # Initialize transformation mode
    curr_mode = "original"

    # Initialize FPS variable
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        rows, cols = frame.shape[:2]

        # Save a copy of original frame
        saved_frame = frame.copy()
        transformed_frame = apply_transformation(saved_frame, curr_mode)

        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time

        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            prev_time = current_time
            frame_count = 0

        # Display FPS on the image
        cv2.putText(frame, f"Origiinal Frame: FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Group 4: Derek Zhang, Yichi Zhang, Yiling Hu, Ziyang Liang", (10, rows - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(transformed_frame, f"Transformed Frame: FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(transformed_frame, "Group 4: Derek Zhang, Yichi Zhang, Yiling Hu, Ziyang Liang", (10, rows - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Stack the original and transformed images horizontally
        combined_frame = cv2.hconcat([frame, transformed_frame])

        # Display the resulting frame
        cv2.imshow('Original and Transformed', combined_frame)

        # Read key for changing the transformation mode
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        curr_mode = set_mode(curr_mode, chr(key))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()