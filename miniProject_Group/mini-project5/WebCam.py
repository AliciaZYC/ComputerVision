# Import necessary packages
import cv2
import time
import numpy as np

# Calculate FPS averaged over the given time interval (default: 1 second).
def calculate_fps(frame_count, prev_time, interval=1.0):
    '''
    Args:
    - frame_count: current count of processed frames
    - prev_time: the last time the FPS calculation was done
    - interval: the time interval over which the FPS is calculated (default is 1 second)

    This function computes the frames per second (FPS) over a given time interval.
    It tracks how many frames have been processed and the time that has passed.

    By checking the elapsed time between the previous time and current time,
    the function calculates the FPS if the interval has passed and resets the frame count and previous time.
    If the interval hasn't passed, it returns None and waits for the next frame.

    Returns:
    - fps: the calculated frames per second
    - prev_time: the updated previous time
    - frame_count: reset or updated frame count
    '''
    current_time = time.time()
    elapsed_time = current_time - prev_time

    if elapsed_time >= interval:
        fps = frame_count / elapsed_time
        prev_time = current_time
        frame_count = 0  # Reset frame count after each interval
        return fps, prev_time, frame_count
    else:
        return None, prev_time, frame_count

# Function to get panorama dimensions and translation homography
def get_panorama_dimensions(image1, image2, H):
    # Get image dimensions
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Get the corners from the images
    corners_image1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    corners_image2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)

    # Transform the corners of image2 to image1's coordinate system
    transformed_corners_image2 = cv2.perspectiveTransform(corners_image2, H)

    # Combine the corners to find the overall bounding box
    all_corners = np.concatenate((corners_image1, transformed_corners_image2), axis=0)

    # Get the min and max x and y coordinates
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Compute the translation homography to shift the panorama so that all pixels are positive
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]], dtype=np.float32)

    # Calculate the size of the panorama
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min

    return (panorama_width, panorama_height), H_translation

# Function to blend two images together
def blend_images(image1, image2):
    '''
    Blends two images together by overlaying non-zero pixels of image2 onto image1.

    Args:
    - image1: The first image (background).
    - image2: The second image to overlay onto the first image.

    Returns:
    - blended_image: The result of blending image2 onto image1.
    '''
    # Create a mask of non-zero pixels in image2
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Create inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Black-out the area of image2 in image1
    image1_bg = cv2.bitwise_and(image1, image1, mask=mask_inv)

    # Take only region of image2 from image2
    image2_fg = cv2.bitwise_and(image2, image2, mask=mask)

    # Add images
    blended_image = cv2.add(image1_bg, image2_fg)

    return blended_image

# Image stitching to create the panorama using SIFT
def stitch_images(frames):
    '''
    Args:
    - frames: list of frames captured from the webcam

    This function stitches multiple frames into a single panorama using SIFT for feature detection and matching.

    Returns:
    - stitched_image: the final stitched panorama
    '''
    stitched_image = frames[0]
    for i in range(1, len(frames)):
        image1 = stitched_image
        image2 = frames[i]

        # Convert images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # Check if descriptors are valid
        if descriptors1 is None or descriptors2 is None:
            print("Failed to compute descriptors.")
            return None

        # Use FLANN-based matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Match descriptors and apply Lowe's ratio test
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Check if we have enough good matches
        MIN_MATCH_COUNT = 10
        if len(good_matches) > MIN_MATCH_COUNT:
            # Extract location of good matches
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography matrix using RANSAC
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                print("Homography could not be computed.")
                return None

            # Get dimensions and translation matrix
            (panorama_width, panorama_height), H_translation = get_panorama_dimensions(image1, image2, H)

            # Warp image1 into the panorama
            result = cv2.warpPerspective(image1, H_translation.dot(H), (panorama_width, panorama_height))

            # Warp image2 into the panorama
            image2_transformed = cv2.warpPerspective(image2, H_translation, (panorama_width, panorama_height))

            # Blend the two images
            stitched_image = blend_images(result, image2_transformed)

        else:
            print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
            return None

    return stitched_image

# Save panorama to disk
def save_panorama(panorama):
    '''
    Args:
    - panorama: the stitched image that will be saved

    This function saves the generated panorama to disk and displays it using OpenCV.

    Returns:
    None
    '''
    if panorama is not None:
        cv2.imshow("Panorama", panorama)
        save_path = "panorama.jpg"
        cv2.imwrite(save_path, panorama)
        print(f"Panorama saved to {save_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def crop_black_area(stitched_image):
    '''
    Args:
    - stitched_image: the stitched panorama image with black areas

    This function crops out the black areas from the panorama image to focus only on the non-black regions.

    Returns:
    - cropped_image: the cropped image without black borders
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)

    # Create a mask where the non-black pixels are set to 1
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the non-black areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        # Crop the original stitched image
        cropped_image = stitched_image[y:y + h, x:x + w]
        return cropped_image
    else:
        return stitched_image

def main():
    vs = cv2.VideoCapture(0)
    if not vs.isOpened():
        print("Error: Could not open video stream.")
        return
    time.sleep(2.0)

    capturing = False
    captured_frames = []
    prev_time = time.time()
    frame_count = 0
    fps = 0

    capture_interval = 0.5  # Interval between captures in seconds
    last_capture_time = None

    while True:
        ret, frame = vs.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Calculate and display FPS
        fps_text, prev_time, frame_count = calculate_fps(frame_count + 1, prev_time)
        if fps_text is not None:
            fps = fps_text

        display_frame = frame.copy()
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press: 'q' to quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if (capturing):
            cv2.putText(display_frame, "Capturing... Press: 'a' to stop", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Press: 's' to start", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Webcam Feed", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Start capturing frames when 's' is pressed
        if key == ord("s"):
            capturing = True
            last_capture_time = time.time()
            print("Started capturing frames for panorama.")

        # Capture frames if capturing is active
        if capturing:
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                captured_frames.append(frame.copy())
                last_capture_time = current_time
                print(f"Captured frame {len(captured_frames)}")

        # Stop capturing when 'a' is pressed
        if key == ord("a"):
            capturing = False
            print("Stopped capturing frames.")
            break

        # Exit on 'q' press
        if key == ord("q"):
            capturing = False
            print("Exiting without capturing.")
            break

    vs.release()
    cv2.destroyAllWindows()

    if len(captured_frames) > 1:
        print("Stitching frames to create panorama...")
        panorama = stitch_images(captured_frames)

        if panorama is not None:
            # Crop the black parts
            panorama_cropped = crop_black_area(panorama)

            # Save the cropped panorama
            save_panorama(panorama_cropped)
        else:
            print("Stitching failed.")
    else:
        print("Not enough frames to create a panorama.")

# Run the main function
if __name__ == "__main__":
    main()
