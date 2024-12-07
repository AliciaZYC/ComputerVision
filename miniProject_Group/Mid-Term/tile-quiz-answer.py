import cv2
import os
import argparse
import numpy as np


def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
    images = [cv2.imread(os.path.join(folder_path, img)) for img in image_files]
    return images

# added code
def find_keypoints_and_match(img1, img2):
    """
    Find keypoints and match between two images using SIFT.
    """
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches


def stitch_images(images):
    """
    Stitches a list of images into a panorama with improved homography estimation.
    """
    if not images:
        print("No images found in the directory.")
        return None

    images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    Hs = [np.eye(3)]

    for i in range(1, len(images)):
        keypoints1, keypoints2, good_matches = find_keypoints_and_match(images_gray[i - 1], images_gray[i])

        if len(good_matches) < 4:
            print(f"Not enough matches between image {i - 1} and image {i} for homography estimation.")
            return None

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.995)

        if H is None:
            print(f"Homography could not be computed between image {i - 1} and image {i}.")
            return None

        H_total = np.dot(Hs[i - 1], H)
        Hs.append(H_total)

    corners = []
    for i in range(len(images)):
        h, w = images[i].shape[:2]
        corners_image = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners_image, Hs[i])
        corners.append(transformed_corners)

    all_corners = np.concatenate(corners, axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = np.array([[1, 0, -xmin],
                            [0, 1, -ymin],
                            [0, 0, 1]])

    panorama = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.float32)
    panorama_mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.float32)

    for i in range(len(images)):
        H = np.dot(translation, Hs[i])
        warped_image = cv2.warpPerspective(images[i].astype(np.float32), H, (panorama.shape[1], panorama.shape[0]))
        mask = cv2.warpPerspective(np.ones_like(images[i][:, :, 0], dtype=np.float32), H, (panorama.shape[1], panorama.shape[0]))

        panorama += warped_image * mask[..., np.newaxis]
        panorama_mask += mask

    panorama_mask[panorama_mask == 0] = 1.0

    stitched_image = (panorama / panorama_mask[..., np.newaxis]).astype(np.uint8)

    stitched_image = crop_black_borders(stitched_image)

    return stitched_image

def crop_black_borders(image):
    """
    Crop the black borders from the stitched image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y + h, x:x + w]
    return image


def combine_images_into_grid(images, rows=2, cols=2):
    if not images:
        print("No images found in the directory.")
        return None

    max_height, max_width = 150, 150
    resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]

    blank_image = np.zeros_like(resized_images[0])
    while len(resized_images) < rows * cols:
        resized_images.append(blank_image)

    rows_of_images = [np.hstack(resized_images[i * cols:(i + 1) * cols]) for i in range(rows)]
    combined_image = np.vstack(rows_of_images)

    return combined_image


def display_stitched_image(image):
    if image is None:
        return

    cv2.imshow('Stitched Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_stitched_image(image, folder_name):
    if image is not None:
        cv2.imwrite(f"{folder_name}.jpg", image)
        print(f"Stitched image saved as {folder_name}.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Stitch images and display.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Path to the folder containing images')
    args = parser.parse_args()

    images = read_images_from_folder(args.directory)

    stitched_image = stitch_images(images)
    display_stitched_image(stitched_image)

    save_stitched_image(stitched_image, args.directory)

if __name__ == '__main__':
    main()
