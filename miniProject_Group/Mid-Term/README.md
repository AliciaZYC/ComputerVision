# CS5330-Mid-Term-Fall2024-Group4
# Image Stitching using SIFT and Homography

## Description

This project aims to stitch multiple images into a seamless panorama using Python and OpenCV. The primary tasks involve:
1. Detecting key features in the images using the SIFT (Scale-Invariant Feature Transform) algorithm.
2. Matching these features between consecutive images.
3. Estimating a homography matrix to align images correctly.
4. Warping and blending the images together to form a single panorama.
5. Removing any black borders from the final stitched image.

The output is a well-aligned, seamless panorama image created from the input set.

## Instructions to Run the Code

### Prerequisites

Make sure you have Python installed along with the required libraries:
- OpenCV
- NumPy
- argparse (included by default with Python)

To install OpenCV and NumPy, run the following command:

```bash
pip install opencv-python numpy
````

### Running the Code
- Prepare your set of images: 
  - Place the images to be stitched together in a single directory. Ensure the images are in supported formats (e.g., .jpg, .png, .jpeg).
  - The script assumes that all images in the directory should be stitched, so ensure only relevant images are present.
  Use the command below to execute the script, specifying the image directory:

- Copy code
```bash
python3 tile-quiz.py -d path_to_your_image_folder
```
Replace path_to_your_image_folder with the path where your images are stored (e.g., data-1).

- The script will stitch the images and display the final result in a window. You can modify the code to save the output image if needed.
### Example Usage
```bash
python3 tile-quiz.py -d ./data-1
```
## Key Algorithms Explained
### Keypoint Matching Algorithm
The SIFT (Scale-Invariant Feature Transform) algorithm is used to detect unique keypoints in each image. Keypoints are distinctive image features that remain robust across various transformations like scaling or rotation. Here's how the matching is performed:

- Keypoint Detection: SIFT identifies a set of keypoints and their descriptors for each image.
- Feature Matching: FLANN (Fast Library for Approximate Nearest Neighbors) is employed to match these descriptors between consecutive images. A two-step process, known as Lowe's ratio test, is used to filter out unreliable matches. This ensures only the most significant and reliable matches are used for alignment.
### Stitching Method
The stitching process includes several stages:

- Homography Calculation:
Using the matched keypoints, a homography matrix is computed via the RANSAC (Random Sample Consensus) method. This matrix determines the perspective transformation required to align one image to another.
- Image Warping and Feather Blending:
Each image is warped using the cumulative homographies to align them in a common perspective. Feather blending is used to merge overlapping regions smoothly, minimizing visible seams.
- Cropping Black Borders:
After stitching, any black regions around the output image are cropped. This is done by thresholding the grayscale image and identifying the bounding box of non-black content.

