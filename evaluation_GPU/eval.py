#python eval.py --trained_model yolo_v2_72.2.pth --input_size 416 --cuda
import torch
import numpy as np
from PIL import Image, ImageDraw
from data import BaseTransform, VOC_CLASSES
import argparse
from models.yolo import myYOLO

parser = argparse.ArgumentParser(description='YOLO Detector Evaluation')
parser.add_argument('--trained_model', type=str, default='yolo_v2_72.2.pth', help='Trained state_dict file path to open')
parser.add_argument('--input_size', default=416, type=int, help='Input size for the model')
parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA if available')

args = parser.parse_args()

def draw_boxes(image, bboxes, cls_inds, scores, labelmap):
    draw = ImageDraw.Draw(image)
    for bbox, cls_ind, score in zip(bboxes, cls_inds, scores):
        x1, y1, x2, y2 = bbox
        label = labelmap[cls_ind] if cls_ind < len(labelmap) else 'Unknown'
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, y1), f'{label} {score:.2f}', fill='red')
    return image

if __name__ == '__main__':
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    input_size = [args.input_size, args.input_size]

    num_classes = len(VOC_CLASSES)

    # Initialize the model
    net = myYOLO(device, input_size=input_size, num_classes=num_classes, trainable=False)

    try:
        state_dict = torch.load(args.trained_model, map_location=device)
        net.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        exit(1)

    net.eval()
    net.to(device)

    transform = BaseTransform(input_size, (0, 0, 0))
    img_path = '/notebooks/bus.jpg'
    try:
        img = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: The image file {img_path} does not exist.")
        exit(1)

    img_orig = img.copy()
    img, _, _ = transform(img)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = net(img)

    # Check the type and content of detections
    print("Detection output type:", type(detections))
    if isinstance(detections, (list, tuple)) and len(detections) == 3:
        bboxes = detections[0]  # Ensure detections are in numpy format
        scores = detections[1]
        cls_inds = detections[2]
    else:
        print("Invalid detection output.")
        exit(1)

    if len(bboxes) == 0:
        print("No detections found.")
        img_orig.save('/notebooks/bus_detected.jpg')
        exit(1)

    img_with_boxes = draw_boxes(img_orig, bboxes, cls_inds, scores, VOC_CLASSES)
    img_with_boxes.save('/notebooks/bus_detected.jpg')

    print('Image saved as /notebooks/bus_detected.jpg')
