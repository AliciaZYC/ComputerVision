# USAGE: python WebCamSave.py -f video_file_name -o out_video.avi
# USAGE: python3 WebCamSave.py -f lane_test1.mp4 -o out_lane_test1.avi
# USAGE: python3 WebCamSave.py -f lane_test2.mp4 -o out_lane_test2.avi
# USAGE: python3 WebCamSave.py -f lane_test3.mp4 -o out_lane_test3.avi
from collections import deque

# import the necessary packages
import cv2
import numpy as np
import time
import os
import argparse
import random


# Buffers to store lane lines over multiple frames
left_line_buffer = deque(maxlen=30)
right_line_buffer = deque(maxlen=30)


def color_filter(image):
    """
    Apply a color filter to isolate white colors in the input image.

    This function converts the input image to HSV color space and creates a mask to filter out
    all colors that do not fall within the white color range (high value, low saturation).

    Args:
        image (ndarray): The input image in BGR format.

    Returns:
        ndarray: The filtered image, where only white pixels are retained.
    """
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for white color in HSV
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 25, 255])

    # Create a mask for white color
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Apply the mask to the image to keep only white regions
    return cv2.bitwise_and(image, image, mask=mask_white)


def region_of_interest(image):
    """
    Apply a mask to keep only the region of interest in the image, typically where lane lines are present.

    The function creates a polygonal mask over the region of interest and applies it to the input image.
    The region is usually the lower half of the image and tapering towards the center.

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: The masked image with only the region of interest visible.
    """
    # Get the height and width of the image
    h, w = image.shape[:2]

    # Create a black mask of the same size as the input image
    mask = np.zeros_like(image)

    # Define the polygon (trapezoid) covering the region of interest
    polygon = np.array([[(0, h), (w, h), (int(w * 0.8), int(h * 0.4)), (int(w * 0.2), int(h * 0.4))]])

    # Fill the polygon on the mask
    cv2.fillPoly(mask, polygon, 255)

    # Apply the mask to the image to keep only the region of interest
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def line_filter_pipeline(image, lines):
    """
    Process and filter the detected lines using slope filtering and RANSAC.

    This function takes the input image and lines detected from a line detection algorithm (like Hough Transform),
    filters them based on slope, and applies RANSAC to remove outliers. It also keeps track of the smoothed lane lines
    by averaging over a buffer of previously detected lines.

    Args:
        image (ndarray): The input image (used for visualization if needed).
        lines (list): A list of lines detected in the image.

    Returns:
        tuple: The smoothed left and right lane lines.
    """
    # Filter the lines based on their slope to remove horizontal or irrelevant lines
    slope_filtered_lines = line_slope_filter(lines)

    # Apply RANSAC to remove outliers and get the most likely lane lines
    ransac_filtered_lines,_ = RANSAC(slope_filtered_lines)

    # Find the best-matching left and right lane lines from the filtered lines
    left_fit, right_fit = find_best_match_line(ransac_filtered_lines)

    # Add the newly detected lines to the buffers (for smoothing)
    add_to_buffer(left_line_buffer, left_fit)
    add_to_buffer(right_line_buffer, right_fit)

    # Smooth the lines by averaging over the buffer of previous frames
    smooth_left_line = average_lines(left_line_buffer)
    smooth_right_line = average_lines(right_line_buffer)

    return smooth_left_line, smooth_right_line


def line_slope_filter(lines):
    """
    Filters the lines based on their slope, removing nearly horizontal or vertical lines.

    This function calculates the slope of each line and filters out lines that are close to vertical
    (undefined slope) or horizontal (slope near zero), as these are not likely to represent lane lines.

    Args:
        lines (list): A list of detected lines, where each line is represented as (x1, y1, x2, y2).

    Returns:
        list: A list of slope-filtered lines.
    """
    slope_filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Avoid division by zero for vertical lines
            if abs(x1 - x2) <= 1e-6:
                continue
            # Calculate the slope of the line
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            # Filter out lines with small slopes (near horizontal)
            if -0.3 < slope < 0.3:
                continue
            slope_filtered_lines.append(line)

    return slope_filtered_lines


def RANSAC(lines, distance_threshold=10, max_iterations=100):
    """
    Applies the RANSAC algorithm to find the best set of inlier lines that intersect at a common point.

    RANSAC is used to identify the intersection of lines that best fit a model, while ignoring outliers.
    This implementation randomly samples pairs of lines, computes their intersection, and counts the number
    of inlier lines that are within a certain distance of that intersection point.

    Args:
        lines (list): A list of detected lines, where each line is represented as (x1, y1, x2, y2).
        distance_threshold (int): The maximum distance a line can be from the intersection to be considered an inlier.
        max_iterations (int): The maximum number of iterations to run RANSAC.

    Returns:
        tuple: The best set of inlier lines and their intersection point.
    """
    best_inliers = []
    best_intersection = None
    if len(lines) < 2:
        return best_inliers, best_intersection
    for _ in range(max_iterations):
        # Randomly select two lines from the set of lines
        line1, line2 = random.sample(lines, 2)

        # Compute the intersection point of the two lines
        intersection = line_intersection(line1[0], line2[0])
        if intersection is None:
            continue  # Skip if lines are parallel

        # Find lines that are close to the intersection (inliers)
        inliers = []
        for line in lines:
            distance = line_point_distance(line[0], intersection)
            if distance < distance_threshold:
                inliers.append(line[0])

        # Update the best inliers and intersection if this set is better
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_intersection = intersection

    return best_inliers, best_intersection


def find_best_match_line(lines, slope_diff_threshold=0.15, min_dis_threshold=30):
    """
    Finds the best matching left and right lane lines from a list of candidate lines.

    This function sorts the lines into left and right lanes based on their slope, then selects the best matches
    by comparing the slopes and distances between lines. The left and right lines with the most consistent slopes
    are returned as the best match.

    Args:
        lines (list): A list of detected lines.
        slope_diff_threshold (float): The maximum allowable difference in slope between two lines to consider them a match.
        min_dis_threshold (int): The minimum distance between two lines for them to be considered a match.

    Returns:
        list: A list containing the best-matched left and right lane lines.
    """
    if lines is None:
        return [], []

    left_match = []
    right_match = []

    for line in lines:
        print(f'Hi, {line}')
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1)
        if slope > 0:
            right_match.append((line, slope))  # Right lane lines have positive slope
        elif slope < 0:
            left_match.append((line, slope))  # Left lane lines have negative slope

    # Sort left and right matches based on their slopes
    left_match.sort(key=lambda l: l[1])  # Sort by slope (ascending) for left lane
    right_match.sort(key=lambda l: l[1], reverse=True)  # Sort by slope (descending) for right lane

    left_best_match = []
    right_best_match = []

    # Select the best left lane line based on slope and distance thresholds
    if len(left_match) > 0:
        left_match_line = left_match[0]
        left_best_match = [line[0] for line in left_match if
                           left_match_line[1] - line[1] <= slope_diff_threshold and line_distance(left_match_line,
                                                                                                  line) < min_dis_threshold]

    # Select the best right lane line based on slope and distance thresholds
    if len(right_match) > 0:
        right_match_line = right_match[0]
        right_best_match = [line[0] for line in right_match if
                            line[1] - right_match_line[1] <= slope_diff_threshold and line_distance(right_match_line,
                                                                                                    line) < min_dis_threshold]

    return left_best_match, right_best_match


def add_to_buffer(buffer, lines):
    """
    Adds the newly detected lines to a buffer for line smoothing.

    The buffer is used to store the detected lines over multiple frames and smooth out noise or fluctuations
    by averaging the lines across the buffer.

    Args:
        buffer (list): The buffer where detected lines are stored.
        lines (list): The newly detected lines to add to the buffer.
    """
    if lines is not None:
        for line in lines:
            buffer.append(line)


def average_lines(buffer):
    """
    Averages the lines stored in the buffer to produce a smooth line.

    This function takes the average of the lines in the buffer to produce a single, stable line that reduces
    noise from frame-to-frame fluctuations in detection.

    Args:
        buffer (list): The buffer containing detected lines.

    Returns:
        ndarray: The averaged line.
    """
    if len(buffer) > 0:
        return np.mean(buffer, axis=0).astype(int)  # Calculate the average of the lines
    return None


def draw_lines(image, left_line, right_line):
    """
    Draws the detected lane lines on the input image.

    This function draws the detected lines on the image, using the intersection point of the left and right lanes
    to extend the lines accurately.

    Args:
        image (ndarray): The input image.
        lines (list): A list of detected lane lines to draw.

    Returns:
        ndarray: The image with the lane lines drawn on it.
    """
    # Find the intersection of the left and right lane lines
    if left_line is not None and right_line is not None:
        intersection = line_intersection(left_line, right_line)

        # Draw each lane line on the image
        x1, y1, x2, y2 = make_line_coordinates(image, left_line, intersection)[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10)  # Red color for lane lines

        x1, y1, x2, y2 = make_line_coordinates(image, right_line, intersection)[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10)  # Red color for lane lines
    return image


def make_line_coordinates(image, line, intersection):
    """
    Generates line coordinates based on the detected line and intersection point.

    This function calculates the coordinates for the lane line, extending it from the bottom
    of the image to a point near the intersection of the left and right lanes, ensuring that
    the lane line is drawn correctly with respect to its slope and intercept.

    Args:
        image (ndarray): The input image where the lines will be drawn.
        line (list): The detected lane line represented as (x1, y1, x2, y2).
        intersection (tuple): The intersection point of the left and right lane lines.

    Returns:
        list: A list containing the coordinates [x1, y1, x2, y2] for the lane line.
    """
    h, w = image.shape[:2]
    if line is None:
        return None

    # Unpack the coordinates of the detected line
    x1, y1, x2, y2 = line
    ix, iy = intersection

    # Fit a line (slope and intercept) to the two points
    slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

    # Set y1 to the bottom of the image
    y1 = h
    # Calculate x1 based on the intercept and slope
    x1 = int((y1 - intercept) / slope)

    # Set y2 as a point close to the intersection, but within the image bounds
    y2 = int(iy + h * 0.3)
    if h * 0.8 < y2:
        y2 = int(h * 0.8)
    # Calculate x2 based on the intercept and slope
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]


def line_intersection(line1, line2):
    """
    Calculates the intersection point of two lines.

    This function checks whether two lines are parallel and, if not, computes the intersection
    point of the two lines using basic line equation math.

    Args:
        line1 (tuple): The first line represented as (x1, y1, x2, y2).
        line2 (tuple): The second line represented as (x3, y3, x4, y4).

    Returns:
        tuple: The intersection point (x, y) if the lines intersect, or None if the lines are parallel.
    """
    # Unpack the coordinates of the two lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate the denominator for intersection formula
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # Lines are parallel

    # Calculate the x and y coordinates of the intersection
    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return (intersect_x, intersect_y)


def line_point_distance(line, point):
    """
    Calculates the perpendicular distance between a line and a point.

    This function uses the formula for the distance between a point and a line segment.

    Args:
        line (tuple): The line represented as (x1, y1, x2, y2).
        point (tuple): The point represented as (px, py).

    Returns:
        float: The shortest distance between the point and the line.
    """
    # Unpack the coordinates of the line and point
    x1, y1, x2, y2 = line
    px, py = point

    # Calculate the numerator for the distance formula
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)

    # Calculate the denominator for the distance formula
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    # Return the calculated distance
    return numerator / denominator


def line_distance(line1, line2):
    """
    Calculates the perpendicular distance between two approximately parallel lines.

    This function assumes the two lines are nearly parallel, based on their slopes, and calculates
    the distance between them using the formula for the distance between two parallel lines.

    Args:
        line1 (tuple): The first line represented as (x1, y1, x2, y2).
        line2 (tuple): The second line represented as (x3, y3, x4, y4).

    Returns:
        float: The perpendicular distance between the two lines.
    """
    # Unpack the coordinates of the two lines
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # Calculate the numerator for the distance formula
    numerator = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1)

    # Calculate the denominator for the distance formula
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    # Return the calculated distance
    return numerator / denominator


def main():
    """
    Main function to capture video input, process it for lane detection, and display the output.

    This function allows for either video file input or camera feed input. It processes each frame
    to detect lane lines using techniques such as color filtering, edge detection, and the Hough Line Transform.
    Detected lane lines are then drawn on the image and displayed in real-time.

    The output can also be saved to a file, if specified. The user can pause and resume the detection process
    or quit by pressing specific keys.

    Command-line Arguments:
        -f, --file: Path to the input video file (optional). If not provided, the camera feed will be used.
        -o, --out: Name of the output video file where the processed video will be saved (optional).

    Key Controls:
        - Press 's' to pause/resume the detection.
        - Press 'q' to quit the program.

    Returns:
        None
    """
    # Set up argument parser for video file input and output options
    parser = argparse.ArgumentParser(description="Video file path or camera input")
    parser.add_argument("-f", "--file", type=str, help="Path to the video file")
    parser.add_argument("-o", "--out", type=str, help="Output video file name")

    args = parser.parse_args()

    # Open video file or capture from the camera
    if args.file:
        vs = cv2.VideoCapture(args.file)
    else:
        vs = cv2.VideoCapture(0)  # 0 is the default camera device

    # Allow time for the camera to initialize
    time.sleep(2.0)

    # Get the default resolution of the video
    width = int(vs.get(3))  # Video frame width
    height = int(vs.get(4))  # Video frame height

    # Set up video writer to save the output video if specified
    out_filename = args.out
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

    paused = False  # Flag to toggle pause/resume functionality

    while True:
        # Check if processing is paused
        if not paused:
            # Read the next frame from the video stream
            ret, frame = vs.read()
            if not ret:
                break

            # Apply color filtering to isolate white lane lines
            filtered_image = color_filter(frame)

            # Convert the filtered image to grayscale
            gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to smooth the image and reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Perform Canny edge detection to identify edges in the image
            edges = cv2.Canny(blurred, 50, 150)

            # Apply region of interest mask to focus on the road lanes
            cropped_edges = region_of_interest(edges)

            # Detect lines using Hough Line Transform
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50,
                                    minLineLength=50, maxLineGap=200)

            # Process the detected lines using the pipeline
            left_line, right_line = line_filter_pipeline(frame, lines)

            # Draw the detected lane lines on the frame
            line_image = draw_lines(frame, left_line, right_line)

            # Combine the original frame with the lane lines
            combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

            # Display the processed frame with lane detection
            display_image = line_image.copy()

            # If output is specified, write the processed frame to the video file
            if args.out:
                out.write(line_image)

            # Display text instructions on the frame
            cv2.putText(display_image, "Lane Detecting......", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_image, "Press 's' to Pause", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(display_image, "Press 'q' to Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Lane Detection", display_image)

        else:
            # Display paused message when detection is paused
            display_image = line_image.copy()
            cv2.putText(display_image, "Paused......", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_image, "Press 's' to Resume", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(display_image, "Press 'q' to Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Lane Detection", display_image)

        # Capture keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Toggle pause/resume when 's' is pressed
            paused = not paused

        elif key == ord('q'):  # Quit the program when 'q' is pressed
            break

    # Release the video stream and writer, and close windows
    vs.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()