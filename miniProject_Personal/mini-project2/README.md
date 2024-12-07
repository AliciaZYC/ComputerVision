# CamTest.py

This Python script uses OpenCV to capture live video from a webcam and provides various functionalities such as cropping, resizing, blurring, adding text, drawing boxes, and more.

## Usage

To run the script, execute the following command:

```bash
cd .../mini-project2
python3 CamTest.py

```

### Keyboard Controls

Once the video feed starts, you can interact with it by pressing the following keys:

| Key | Action Description                                              |
| --- | --------------------------------------------------------------- |
| `q` | Quit the video feed and close the window.                       |
| `c` | Crop the video frame.                                           |
| `r` | Resize the video frame.                                         |
| `b` | Apply a blur effect to the video frame.                         |
| `a` | Draw a green box on the video frame.                            |
| `t` | Add text "Yichi Zhang" to the video frame.                      |
| `g` | Apply thresholding (binary black-and-white) to the video frame. |
| `n` | Detect and draw contours on the video frame.                    |

### Functionalities Explained

1. **Crop Video (`c` or `C`):**
   - This option crops the frame, displaying a central portion of the video (from 20% to 80% in both height and width).
2. **Resize Video (`r` or `R`):**
   - This resizes the video to a fixed width of 300 pixels, maintaining the aspect ratio by adjusting the height accordingly.
3. **Blur Video (`b` or `B`):**
   - A Gaussian blur effect is applied to the current frame to create a smoothing effect.
4. **Add a Box (`a` or `A`):**
   - Draws a green rectangle in the middle of the video frame.
5. **Add Text (`t` or `T`):**
   - Adds the text "Yichi Zhang" to the top left corner of the video frame.
6. **Thresholding (`g` or `G`):**
   - Converts the frame to grayscale and applies binary thresholding, turning it into a black-and-white image.
7. **Detect and Draw Contours (`n` or `N`):**
   - Detects external contours from the thresholded image and draws these contours on the original frame in purple.

## Example Output

You can see the sample_test_video.mp4 in the same folder for reference.
