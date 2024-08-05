
import yaml
import argparse
import cv2
import numpy as np
import re
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data

CLASSES = yaml_load(check_yaml("coco128.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.
    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main(model_path, input_image):
    """
    Main function to load YOLO model, perform inference, draw bounding boxes, and display the output image.
    Args:
        model_path (str): Path to the YOLO model.
        input_image (str): Path to the input image.
    Returns:
        list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
    """
    # Load the YOLO model
    model = YOLO(model_path)  # Ensure model_path is correctly provided here

    # Read the input image
    original_image = cv2.imread(input_image)
    [height, width, _] = original_image.shape

    # Perform inference
    results = model.predict(source=input_image, imgsz=640)

    detections = []
    for result in results:
        # Extract bounding boxes, confidences, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                "class_id": class_id,
                "class_name": CLASSES[class_id],
                "confidence": float(score),
                "box": [x1, y1, x2 - x1, y2 - y1]
            })
            draw_bounding_box(original_image, class_id, float(score), x1, y1, x2, y2)

    # Save the image with bounding boxes
    cv2.imwrite("output_image.jpg", original_image)
    print("Image saved as output_image.jpg")
    
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt", help="Path to your YOLO model.")
    parser.add_argument("--img", default="bus.jpg", help="Path to input image.")
    args = parser.parse_args()
    main(args.model, args.img)
