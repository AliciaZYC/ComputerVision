import cv2
import os
import argparse
import numpy as np
from collections import defaultdict

def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
    images = [cv2.imread(os.path.join(folder_path, img)) for img in image_files]
    return images, image_files

def find_keypoints_and_match(img1, img2):
    """
    Find keypoints and match between two images using SIFT.
    """
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        return None, None, []

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    except cv2.error:
        return None, None, []

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches

def compute_pairwise_matches(images):
    num_images = len(images)
    match_scores = np.zeros((num_images, num_images), dtype=np.int32)
    matches_dict = defaultdict(dict)
    for i in range(num_images):
        for j in range(i+1, num_images):
            keypoints1, keypoints2, good_matches = find_keypoints_and_match(images[i], images[j])
            match_count = len(good_matches)
            match_scores[i, j] = match_count
            match_scores[j, i] = match_count
            if match_count > 10:
                matches_dict[i][j] = (keypoints1, keypoints2, good_matches)
                matches_dict[j][i] = (keypoints2, keypoints1, good_matches)
    return match_scores, matches_dict

def build_image_graph(match_scores):
    num_images = match_scores.shape[0]
    graph = defaultdict(list)
    for i in range(num_images):
        sorted_indices = np.argsort(-match_scores[i])
        for j in sorted_indices:
            if i != j and match_scores[i, j] > 10:
                graph[i].append(j)
    return graph

def arrange_images_into_grid(graph):
    positions = {}
    used = set()

    start_img = next(iter(graph.keys()))
    positions[start_img] = (0, 0)
    used.add(start_img)
    queue = [start_img]

    while queue:
        current = queue.pop(0)
        current_pos = positions[current]

        neighbors = graph[current]
        for neighbor in neighbors:
            if neighbor in used:
                continue

            potential_positions = [
                (current_pos[0] + 1, current_pos[1]),  # Right
                (current_pos[0], current_pos[1] + 1),  # Down
                (current_pos[0] - 1, current_pos[1]),  # Left
                (current_pos[0], current_pos[1] - 1)   # Up
            ]
            for pos in potential_positions:
                if pos not in positions.values():
                    positions[neighbor] = pos
                    used.add(neighbor)
                    queue.append(neighbor)
                    break

    return positions

def create_grid(positions, images):
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    min_x, min_y = min(xs), min(ys)

    normalized_positions = {}
    for img_idx, (x, y) in positions.items():
        normalized_positions[img_idx] = (x - min_x, y - min_y)

    grid_width = max(xs) - min_x + 1
    grid_height = max(ys) - min_y + 1

    grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]

    for img_idx, (x, y) in normalized_positions.items():
        grid[y][x] = images[img_idx]

    return grid

def stitch_grid(grid):
    stitched_rows = []
    for row in grid:
        row_images = [img for img in row if img is not None]
        if len(row_images) == 0:
            continue
        stitched_row = stitch_images(row_images, direction='horizontal')
        if stitched_row is not None:
            stitched_rows.append(stitched_row)

    if len(stitched_rows) == 0:
        return None

    stitched_image = stitched_rows[0]
    for i in range(1, len(stitched_rows)):
        stitched_image = stitch_images([stitched_image, stitched_rows[i]], direction='vertical')
        if stitched_image is None:
            return None

    return stitched_image

def stitch_images(images, direction='horizontal'):
    """
    Stitches a list of images into a panorama with improved homography estimation.
    """
    if not images:
        print("No images to stitch.")
        return None

    images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    Hs = [np.eye(3)]

    for i in range(1, len(images)):
        keypoints1, keypoints2, good_matches = find_keypoints_and_match(images_gray[i - 1], images_gray[i])

        if len(good_matches) < 4:
            print(f"Not enough matches between images for homography estimation.")
            return None

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        if direction == 'horizontal':
            H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
        else:
            H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        if H is None:
            print(f"Homography could not be computed between images.")
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

def display_stitched_image(image):
    if image is None:
        print("No stitched image to display.")
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
    parser = argparse.ArgumentParser(description='Stitch images into a grid.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Path to the folder containing images')
    args = parser.parse_args()

    images, image_files = read_images_from_folder(args.directory)

    if len(images) < 2:
        print("Need at least 2 images to stitch.")
        return

    print("Computing pairwise matches...")
    match_scores, matches_dict = compute_pairwise_matches(images)

    print("Building image graph...")
    graph = build_image_graph(match_scores)

    print("Arranging images into grid...")
    positions = arrange_images_into_grid(graph)

    if len(positions) != len(images):
        print("Could not arrange all images into grid.")
        return

    print("Creating image grid...")
    grid = create_grid(positions, images)

    print("Stitching image grid...")
    stitched_image = stitch_grid(grid)

    display_stitched_image(stitched_image)

    save_stitched_image(stitched_image, args.directory)

if __name__ == '__main__':
    main()
