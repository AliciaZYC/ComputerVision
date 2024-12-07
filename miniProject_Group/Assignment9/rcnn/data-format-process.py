import os
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
import cv2
#Step 1: Convert .XML to .CSV
# Define folders
image_folder = "test"  # Folder containing the original images
xml_folder = "test"  # Folder containing the XML files
output_csv_folder = "test/annotations"  # Folder to save CSV files

# Create output folder if it doesn't exist
os.makedirs(output_csv_folder, exist_ok=True)

# Loop through each XML file in the folder
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_folder, xml_file)

        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Initialize a list to store bounding box data
        csv_data = []

        # Loop through each object in the XML
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")

            # Extract bounding box coordinates
            x1 = int(bndbox.find("xmin").text)
            y1 = int(bndbox.find("ymin").text)
            x2 = int(bndbox.find("xmax").text)
            y2 = int(bndbox.find("ymax").text)

            # Append bounding box coordinates as a string in the format "x1 y1 x2 y2"
            csv_data.append(f"{x1} {y1} {x2} {y2}")

        # Prepare the first row with the total number of objects
        num_objects = len(csv_data)
        csv_data.insert(0, str(num_objects))  # Insert the count at the beginning of the list

        # Convert the list to a DataFrame
        df = pd.DataFrame(csv_data)

        # Define the CSV filename
        csv_filename = os.path.splitext(xml_file)[0] + ".csv"
        csv_path = os.path.join(output_csv_folder, csv_filename)

        # Save DataFrame to CSV without headers or index
        df.to_csv(csv_path, index=False, header=False)

        print(f"Converted {xml_file} to {csv_filename}")

# Step 2: Rename Images to reable names
csv_folder = "test/annotations"

# List all images in the image folder
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])

# Loop through each image and rename it along with its corresponding CSV file
for i, image_file in enumerate(image_files, start=1):
    # Define new base name
    new_base_name = f"wine_bottle{i}"

    # Get the current image and CSV paths
    image_path = os.path.join(image_folder, image_file)
    csv_path = os.path.join(csv_folder, os.path.splitext(image_file)[0] + ".csv")

    # Define new image and CSV paths
    new_image_name = f"{new_base_name}{os.path.splitext(image_file)[1]}"  # Preserve file extension
    new_image_path = os.path.join(image_folder, new_image_name)

    new_csv_name = f"{new_base_name}.csv"
    new_csv_path = os.path.join(csv_folder, new_csv_name)

    # Rename image
    os.rename(image_path, new_image_path)
    print(f"Renamed {image_file} to {new_image_name}")

    # Rename corresponding CSV file if it exists
    if os.path.exists(csv_path):
        os.rename(csv_path, new_csv_path)
        print(f"Renamed {os.path.basename(csv_path)} to {new_csv_name}")
    else:
        print(f"Warning: CSV file for {image_file} not found")

print("Renaming complete!")

#Optional: Change Size
# Paths to your images and annotations directories
image_dir = "test/winebottle"
annotation_dir = "test/annotations"
output_image_dir = "test/resized_images"
output_annotation_dir = "test/resized_annotations"

# Set new dimensions
new_width, new_height = 640, 640

# Ensure output directories exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

# Iterate over each annotation file
for annotation_file in os.listdir(annotation_dir):
    # Get the corresponding image filename
    image_filename = os.path.splitext(annotation_file)[0] + ".jpg"
    image_path = os.path.join(image_dir, image_filename)
    annotation_path = os.path.join(annotation_dir, annotation_file)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read {image_filename}. Skipping.")
        continue

    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height))
    output_image_path = os.path.join(output_image_dir, image_filename)
    cv2.imwrite(output_image_path, resized_image)

    # Load annotations
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    # First line is the number of objects
    num_objects = int(lines[0].strip())
    updated_annotations = [f"{num_objects}\n"]

    # Process each bounding box
    for line in lines[1:]:
        # Parse bounding box coordinates from the format "x1 y1 x2 y2"
        x1, y1, x2, y2 = map(int, line.strip().split())

        # Scale bounding box coordinates
        x1 = int(x1 * new_width / original_width)
        y1 = int(y1 * new_height / original_height)
        x2 = int(x2 * new_width / original_width)
        y2 = int(y2 * new_height / original_height)

        # Add updated bounding box to list in the format "x1 y1 x2 y2"
        updated_annotations.append(f"{x1} {y1} {x2} {y2}\n")

    # Save the updated annotations to a new file
    output_annotation_path = os.path.join(output_annotation_dir, annotation_file)
    with open(output_annotation_path, 'w') as f:
        f.writelines(updated_annotations)

print("Images resized and bounding boxes updated.")