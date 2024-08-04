import os
import cv2
import torch
import pandas as pd
import time

from detectron2_for_evaluation.config import get_cfg
from detectron2_for_evaluation.engine.defaults import DefaultPredictor
from detectron2_for_evaluation import model_zoo

def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Check if CUDA (GPU) is available, else use CPU
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cuda'
    else:
        cfg.MODEL.DEVICE = 'cpu'

    return DefaultPredictor(cfg)

def get_predictions(predictor, image_path):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    return outputs["instances"].to("cpu")  # Ensure this is set to CPU

def load_ground_truth(csv_file):
    df = pd.read_csv(csv_file)
    ground_truth = {}
    for _, row in df.iterrows():
        image_name = row["image_name"]
        xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        if image_name not in ground_truth:
            ground_truth[image_name] = []
        ground_truth[image_name].append((xmin, ymin, xmax, ymax))
    return ground_truth

def check_overlap(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Check if there is any overlap
    if x1 < x2_p and x2 > x1_p and y1 < y2_p and y2 > y1_p:
        return True
    return False

def compute_accuracy(ground_truth, predictor, image_dir):
    start_time = time.time()  # Start timing
    
    total_true_positive = 0
    total_ground_truth = 0
    
    for image_name, gt_boxes in ground_truth.items():
        image_path = os.path.join(image_dir, image_name + ".jpg")
        predictions = get_predictions(predictor, image_path)
        
        pred_boxes = predictions.pred_boxes.tensor.numpy()
        
        # Check if any predictions overlap with ground truth
        for gt_box in gt_boxes:
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
            match_found = False
            for pred_box in pred_boxes:
                if check_overlap((gt_xmin, gt_ymin, gt_xmax, gt_ymax), tuple(pred_box)):
                    match_found = True
                    break
            if match_found:
                total_true_positive += 1
            total_ground_truth += 1
    
    accuracy = total_true_positive / total_ground_truth if total_ground_truth > 0 else 0
    
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    return accuracy
    #print(f"Accuracy: {accuracy:.2f}")
    #print(f"Time taken: {elapsed_time:.2f} seconds")

def get_accuracy_option1(path_to_data):
    csv_file_path = "bounding_boxes_with_labels.csv"  # path to your CSV file
    image_directory = path_to_data+"/images"  # path to your images directory

    predictor = setup_predictor()

    # Load ground truth and compute accuracy
    ground_truth = load_ground_truth(csv_file_path)
    return compute_accuracy(ground_truth, predictor, image_directory)


def detectron2_evaluation(path_to_data):
    # make_coco_dataset(path_to_data)
    return (get_accuracy_option1(path_to_data))#,get_accuracy_option2(path_to_data))