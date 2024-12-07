import cv2
import time
import numpy as np


def draw_matches_with_scores(img1, kp1, img2, kp2, matches, max_matches=10):
    """
    Draws the matches between two images with their scores.

    Parameters:
    - img1: First image.
    - kp1: Keypoints of the first image.
    - img2: Second image.
    - kp2: Keypoints of the second image.
    - matches: List of DMatch objects containing the matched keypoints.
    - max_matches: Maximum number of matches to display.

    Returns:
    - img_match: Image with matches drawn.
    """

    # Sort the matches based on their distance (score)
    matches = sorted(matches, key=lambda x: x.distance)

    # Select the top 'max_matches' matches
    display_matches = matches[:max_matches]

    # Draw matches with scores
    img_match = cv2.drawMatches(
        img1, kp1, img2, kp2, display_matches, None,
        matchColor=(0, 255, 0),  # Color of the matching lines
        singlePointColor=(255, 0, 0),  # Color of the keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Add the score (distance) as text on the matches
    for i, match in enumerate(display_matches):
        pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        score_text = f"Score: {match.distance:.2f}"

        # Put text on the matching line midpoint
        mid_point = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
        cv2.putText(img_match, score_text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return img_match


# Method to initialize camera and get its resolution
def initialize_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height


# Method to determine the target resolution for the frames
def get_target_resolution(width1, height1, width2, height2):
    target_width = max(width1, width2)
    target_height = max(height1, height2)
    return target_width, target_height


# Method to capture and resize.py frames from both cameras
def capture_and_resize_frames(cap1, cap2, target_width, target_height):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        return None, None

    frame1 = cv2.resize(frame1, (target_width, target_height))
    frame2 = cv2.resize(frame2, (target_width, target_height))
    return frame1, frame2


# Method to detect keypoints and descriptors using ORB
def detect_keypoints_and_descriptors(orb, gray1, gray2):
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    return keypoints1, descriptors1, keypoints2, descriptors2


# Method to convert frames to grayscale
def convert_to_grayscale(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return gray1, gray2


# Method to match keypoints using BFMatcher
def match_keypoints(bf, descriptors1, descriptors2):
    if descriptors1 is not None and descriptors2 is not None:
        matches = bf.match(descriptors1, descriptors2)
        return matches
    return None


# Calculate FPS averaged over the given time interval (default: 1 second).
def calculate_fps(frame_count, prev_time, interval=1.0):
    current_time = time.time()
    elapsed_time = current_time - prev_time

    if elapsed_time >= interval:
        fps = frame_count / elapsed_time
        prev_time = current_time
        frame_count = 0  # Reset frame count after each interval
        return fps, prev_time, frame_count
    else:
        return None, prev_time, frame_count


def main():
    # Initialize cameras
    cap1, width1, height1 = initialize_camera(0)  # First camera
    cap2, width2, height2 = initialize_camera(1)  # Second camera

    # Set the target resolution
    target_width, target_height = get_target_resolution(width1, height1, width2, height2)

    # Initialize ORB detector and BFMatcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Initialize FPS variables
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        # Capture and resize.py frames
        frame1, frame2 = capture_and_resize_frames(cap1, cap2, target_width, target_height)
        if frame1 is None or frame2 is None:
            break

        # Convert to grayscale
        gray1, gray2 = convert_to_grayscale(frame1, frame2)

        # Detect keypoints and descriptors
        keypoints1, descriptors1, keypoints2, descriptors2 = detect_keypoints_and_descriptors(orb, gray1, gray2)

        # Match descriptors
        matches = match_keypoints(bf, descriptors1, descriptors2)
        if matches:
            matched_img = draw_matches_with_scores(frame1, keypoints1, frame2, keypoints2, matches, max_matches=10)

            # Calculate FPS over the last second
            frame_count += 1
            new_fps, prev_time, frame_count = calculate_fps(frame_count, prev_time)

            if new_fps is not None:
                fps = new_fps  # Update the FPS value once every second

            # Display FPS on the image
            cv2.putText(matched_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the matched image
            cv2.imshow('Matches with Scores', matched_img)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras and close all OpenCV windows
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


# Run the main function
if __name__ == "__main__":
    main()
