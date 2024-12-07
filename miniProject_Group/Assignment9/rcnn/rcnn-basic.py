import os
import cv2
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Model

# Path to dataset (images) and label (csv)
image_path = "dataset/train/winebottle"
label_dir = "dataset/train/annotations"

# Lists to store data
train_images = []
train_labels = []
svm_images = []
svm_labels = []

# IoU calculation
def get_iou(bb1, bb2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return max(0.0, min(1.0, iou))

# Step 1: Running Selective Search on individual images to obtain region proposals (2000 here).
# Enable optimized computation in OpenCV
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

for e, i in enumerate(os.listdir(label_dir)):
    try:
        filename = i.split(".")[0] + ".jpg"
        # print(e, filename)
        image = cv2.imread(os.path.join(image_path, filename))
        df = pd.read_csv(os.path.join(label_dir, i))
        gtvalues = []

        for _, row in df.iterrows():

            x1, y1, x2, y2 = map(int, row.iloc[0].split(" "))

            gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
            # Ground truth image added to SVM data
            timage = image[y1:y2, x1:x2]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
            svm_images.append(resized)
            svm_labels.append([0, 1])

        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = image.copy()
        counter = 0
        falsecounter = 0
        flag = False

        # Step 2: Classifying region proposals as positive and negative examples based on IoU.
        for e, result in enumerate(ssresults):
            if e < 2000 and not flag:
                x, y, w, h = result
                for gtval in gtvalues:
                    iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                    timage = imout[y:y + h, x:x + w]

                    if counter < 30 and iou > 0.7:  # Positive examples for training
                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(1)
                        counter += 1
                    elif falsecounter < 30 and iou < 0.3:  # Negative examples for training
                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(0)
                        falsecounter += 1
                    if falsecounter < 5 and iou < 0.3:  # Negative examples for SVM
                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                        svm_images.append(resized)
                        svm_labels.append([1, 0])

                if counter >= 30 and falsecounter >= 30:
                    flag = True  # Stop if we have enough examples
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

# Conversion of train data into arrays for further training
X_new = np.array(train_images)
Y_new = np.array(train_labels)

# Step 3: Passing every proposal through a pretrained network (VGG16 trained on ImageNet) to output a fixed-size feature vector (4096 here).
vgg = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
for layer in vgg.layers[:-2]:
    layer.trainable = False
x = vgg.get_layer('fc2').output
x = Dense(1, activation='sigmoid')(x)
model = Model(vgg.input, x)
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])
model.summary()
#model.fit(X_new, Y_new, batch_size=32, epochs=3, verbose=1, validation_split=0.05, shuffle=True)
model.fit(X_new, Y_new, batch_size=16, epochs=1, verbose=1, validation_split=0.05, shuffle=True)

# Step 4: Using this feature vector to train an SVM.
x = model.get_layer('fc2').output
Y = Dense(2)(x)
final_model = Model(model.input, Y)
final_model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
final_model.summary()

# Train SVM model
hist_final = final_model.fit(np.array(svm_images), np.array(svm_labels),
                             #batch_size=32, epochs=10, verbose=1,
                             batch_size=16, epochs=2, verbose=1,
                             validation_split=0.05, shuffle=True)
final_model.save('keyboard_model_weights.h5')

# Step 5: Non-maximum Suppression (NMS) to remove redundant overlapping bounding boxes
def non_max_suppression(boxes, overlapThresh):
    """Perform non-maximum suppression on bounding boxes."""
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Step 6: Bounding box regression to refine the positions of the bounding boxes
# Note: For simplicity, we'll skip actual regression model training.

# Plotting loss and analyzing losses
plt.plot(hist_final.history['loss'])
plt.plot(hist_final.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"])
plt.savefig('chart_loss.png')
plt.show()


# Testing on new images
# Path to test dataset (images) and label (csv)
test_image_path = "dataset/test/winebottle"
test_label_dir = "dataset/test/annotations"

# Get a list of image files in the folder
image_files = [f for f in os.listdir(test_image_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Randomly select 10 images from the list
random_images = random.sample(image_files, min(10, len(image_files)))  # In case there are fewer than 10 images

# Loop through each randomly selected image
for image_file in random_images:
    # Load the image
    image_path = os.path.join(test_image_path, image_file)
    image = cv2.imread(image_path)

    # Copy of the image to draw bounding boxes
    imOut = image.copy()

    # Load ground truth bounding boxes from annotations
    annotation_file = os.path.join(test_label_dir, image_file.split(".")[0] + ".csv")
    if os.path.exists(annotation_file):
        df = pd.read_csv(annotation_file)
        for _, row in df.iterrows():
            x1, y1, x2, y2 = map(int, row[0].split(" "))
            # Draw ground truth bounding box in blue
            cv2.rectangle(imOut, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)  # Blue color

    # Initialize Selective Search on the image
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()

    boxes = []

    # Process each region proposed by selective search
    for e, result in enumerate(ssresults):
        if e < 100:  # Limiting to top 50 regions for speed
            x, y, w, h = result
            timage = image[y:y + h, x:x + w]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
            resized = np.expand_dims(resized, axis=0)

            # Predict using the model
            out = final_model.predict(resized)

            # Get confidence score (assuming second value is the score)
            score = out[0][1]
            if score > 0.7:  # Threshold for confidence
                boxes.append([x, y, x + w, y + h, score])

    # Convert list of boxes to numpy array for NMS
    boxes = np.array(boxes)

    # Apply Non-maximum Suppression (NMS)
    nms_boxes = non_max_suppression(boxes, overlapThresh=0.3)

    # Draw predicted bounding boxes on the image
    for box in nms_boxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)  # Green color for predictions

    # Display the image with bounding boxes
    plt.figure()
    plt.imshow(cv2.cvtColor(imOut, cv2.COLOR_BGR2RGB))
    plt.title(f"Detection for {image_file}")
    plt.axis('off')
    plt.show()
