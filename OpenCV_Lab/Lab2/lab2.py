import cv2

# 1. Load Images with OpenCV
def load_images():
    # Load a color image
    img = cv2.imread("cake.jpg", cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Could not load image.")
        return None, None

    # Load the same image in grayscale
    gray_img = cv2.imread("cake.jpg", cv2.IMREAD_GRAYSCALE)

    return img, gray_img

# 2. Display Images with OpenCV
def display_images(img, gray_img):
    # Display the color image
    cv2.imshow("Color Image", img)

    # Display the grayscale image
    cv2.imshow("Grayscale Image", gray_img)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 3. Save Images with OpenCV
def save_images(img):
    # Save the color image with a new name
    cv2.imwrite("lab_2_image.jpg", gray_img)
    print("Image saved as lab_2_image.jpg")

if __name__ == "__main__":
    # Load the images
    img, gray_img = load_images()
    if img is not None:
        # Display the images
        display_images(img, gray_img)

        # Save the image
        save_images(gray_img)
