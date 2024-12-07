import cv2
import numpy as np

def harris_corner_detection(image):
    """Applies Harris Corner Detection on the given image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Red corners
    return image

def sift_detection(image):
    """Applies SIFT keypoint detection on the given image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    img_sift = cv2.drawKeypoints(image, keypoints, None)
    return img_sift

def flann_match(img1, img2):
    """Performs FLANN-based feature matching."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
    return match_img

def main():
    # Load images
    img1 = cv2.imread("image1.jpg")
    img2 = cv2.imread("image2.jpg")

    # Harris Corner Detection
    harris_img = harris_corner_detection(img1.copy())
    cv2.imshow("Harris Corners", harris_img)

    # SIFT Keypoint Detection
    sift_img = sift_detection(img1.copy())
    cv2.imshow("SIFT Keypoints", sift_img)

    # FLANN-based Feature Matching
    matched_img = flann_match(img1, img2)
    cv2.imshow("FLANN Matching", matched_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
