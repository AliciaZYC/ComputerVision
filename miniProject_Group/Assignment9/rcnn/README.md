# Assignment9: RCNN Object Detection

## Project Overview
This repository contains code, dataset, and trained model for object detection using RCNN. The primary objective is to detect specific classes in images using a model trained with manually labeled data.

### Dataset
- **Classes**: "wine bottle"
- **Number of Images**: 158 images for training, 6 for testing
- **Collection**: Images were basically collected from Roboflow to ensure a diverse dataset covering multiple perspectives, lighting conditions, and backgrounds.
- **Labeling**: Most of the images were already correctly labeled while we manually adjust the image and annotation formats. The labeling format is in CSV, where each file contains bounding box coordinates (`x1 y1 x2 y2`) for objects in each image.

### Screenshots
Below are screenshots demonstrating the model's training progress and performance.

#### Training Loss and Accuracy Graph
![RCNN-BASIC Loss/Accuracy Graph](https://drive.google.com/file/d/15--ULnNTn7g_PMeqtS-pSVg-2vHcOvo4/view?usp=sharing)
![RCNN-LIGHT Loss/Accuracy Graph](https://drive.google.com/file/d/1OieVBkYmmTuOUrbbIqjYL-1zi0XiUO24/view?usp=sharing)

#### Execution Results(pdf version)
The model trained is not accurate enough, maybe due to the small dataset.

![RCNN-BASIC](https://drive.google.com/file/d/19mUyDcXsZAfafhO_b7iR3u6r472sklZW/view?usp=drive_link)
![RCNN-LIGHT](https://drive.google.com/file/d/1i29XgRvz8WBIlNkdVUXb0UvnYAi_fY5q/view?usp=sharing)

#### Videos
![RCNN-BASIC](https://drive.google.com/file/d/1wQ74UPdhDs1dhvzilbiKQ8KyV422d8XD/view?usp=sharing)
![RCNN-LIGHT](https://drive.google.com/file/d/1k03uoOJCm1gt0g6gWcedRqMF4-7w8lEX/view?usp=sharing)

### Improved Parameters(NOTE: fps improved from 0.08 to around 0.35 but still very slow, possibly also due to the limitation of my old laptop)

- **MobileNet instead of VGG16**  
MobileNet is a lightweight, efficient model optimized for mobile and real-time applications. It processes faster than VGG16 with a smaller computational footprint, making it ideal for real-time detection.

- **Resize frame to 320 x 320 before detecting object**  
Resizing frames to 320x320 reduces input size and computational load, enhancing processing speed. This small trade-off in resolution still provides sufficient detail for object detection.

- **Reduce Region of Proposal to 30**  
Limiting region proposals to 30 focuses the model on the most relevant areas, reducing processing time and minimizing false positives by discarding irrelevant regions.

### Code(Colab)
![RCNN-BASIC](https://colab.research.google.com/drive/1rabL7idD0XpjW3Ua4kQZjVBWR3XzL155#scrollTo=nINVr2cLkTTr)
![RCNN-LIGHT](https://colab.research.google.com/drive/15G-mEQxMo6oF_r5ZRNehRY8lcI5LRHno?usp=sharing)

### Dataset
The dataset is available on Google Drive, including all labeled images and CSV files with bounding box annotations.  
[Download Dataset](https://drive.google.com/file/d/1ekbRpQ6xqWrlLY1dHCeNjJPoCgi5CPoz/view?usp=sharing)

### Trained Model
The trained model weights and architecture are saved and available on Google Drive for testing and further evaluation.
[RCNN-BASIC](https://drive.google.com/file/d/16Vk2E_P1bE5BUy-0OZ8qmQj2p4DsH1XZ/view?usp=sharing)
[RCNN-LIGHT](https://drive.google.com/file/d/1oDEk6CrWZVQFi0nPjXQE3WFTQUzh2ZYx/view?usp=sharing)

---

## Execution [NOTE: the training code has never directly run on Laptop, instead Colab was used for the assignment, therefore there might exist name/parameter mismatch]

To run the code, clone this repository and execute `rcnn-basic.py`,`rcnn-light.py`, `WebCamSave_rcnn.py`. Ensure all dependencies are installed as specified in the `requirements.txt` file.

### Sample Code Execution
```bash
python rcnn-basic.py
python rcnn-light.py
python3 WebCamSave_rcnn.py -f test.mp4 -o out_video.avi