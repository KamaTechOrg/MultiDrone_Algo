# Import necessary libraries
import os
import cv2
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logger for Detectron2

# Global variables
Selected_Fold = 2  # 0..2

# Function to get image dimensions
def get_image_dimensions(image_dir, image_name):
    image_path = os.path.join(image_dir, image_name + ".jpg")  # Assuming images are .jpg files
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image {image_path} not found.")
    height, width = image.shape[:2]
    return width, height

# Function to extract bounding boxes from labels
def extract_bounding_boxes(labels_dir, image_dir):
    data = []
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):  # Assuming the bounding box files have .txt extension
            image_name = os.path.splitext(filename)[0]
            width, height = get_image_dimensions(image_dir, image_name)
            with open(os.path.join(labels_dir, filename), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:  # Assuming the format includes class label
                        class_label, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                        xmin = int((x_center - bbox_width / 2) * width)
                        ymin = int((y_center - bbox_height / 2) * height)
                        xmax = int((x_center + bbox_width / 2) * width)
                        ymax = int((y_center + bbox_height / 2) * height)
                        data.append([image_name, int(class_label), xmin, ymin, xmax, ymax])
                    else:
                        print(f"Unexpected format in file: {filename}")
    return data

# Function to create CSV file from labels
def create_csv(labels_dir, image_dir, output_csv):
    data = extract_bounding_boxes(labels_dir, image_dir)
    df = pd.DataFrame(data, columns=["image_name", "class_label", "xmin", "ymin", "xmax", "ymax"])
    df.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' created successfully.")

# Function to generate CSV file for the dataset
def make_csv(path_to_data):
    labels_directory = path_to_data + "/labels"  # Path to labels directory
    image_directory = path_to_data + "/images"  # Path to images directory
    output_csv_file = "bounding_boxes_with_labels.csv"  # Output CSV file name
    create_csv(labels_directory, image_directory, output_csv_file)