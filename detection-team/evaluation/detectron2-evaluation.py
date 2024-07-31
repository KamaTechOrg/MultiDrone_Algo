import os
import cv2
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import pandas as pd
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
import pandas as pd
import numpy as np
import pandas as pd 
import os
import cv2
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import time
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm # progress bar
from tqdm.notebook import tqdm
tqdm.pandas()

from datetime import datetime
import time
import matplotlib.pyplot as plt

import os, json, cv2, random
import skimage.io as io
import copy
from pathlib import Path
from typing import Optional
import json
import matplotlib.pyplot as plt

import ast

from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import itertools

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from glob import glob
import numba
from numba import jit

from pycocotools.coco import COCO
# detectron2
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils


from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset



from detectron2.evaluation.evaluator import DatasetEvaluator
import pycocotools.mask as mask_util
from detectron2.engine import BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer
import torch, torchvision

setup_logger()
Selected_Fold=2 #0..2

def get_image_dimensions(image_dir, image_name):
    image_path = os.path.join(image_dir, image_name + ".jpg")  # assuming images are .jpg files
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image {image_path} not found.")
    height, width = image.shape[:2]
    return width, height

def extract_bounding_boxes(labels_dir, image_dir):
    data = []
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):  # assuming the bounding box files have .txt extension
            image_name = os.path.splitext(filename)[0]
            width, height = get_image_dimensions(image_dir, image_name)
            with open(os.path.join(labels_dir, filename), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:  # assuming the format now includes class label
                        class_label, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                        xmin = int((x_center - bbox_width / 2) * width)
                        ymin = int((y_center - bbox_height / 2) * height)
                        xmax = int((x_center + bbox_width / 2) * width)
                        ymax = int((y_center + bbox_height / 2) * height)
                        data.append([image_name, int(class_label), xmin, ymin, xmax, ymax])
                    else:
                        print(f"Unexpected format in file: {filename}")
    return data

def create_csv(labels_dir, image_dir, output_csv):
    data = extract_bounding_boxes(labels_dir, image_dir)
    df = pd.DataFrame(data, columns=["image_name", "class_label", "xmin", "ymin", "xmax", "ymax"])
    df.to_csv(output_csv, index=False)
    # print(f"CSV file '{output_csv}' created successfully.")

def make_csv(path_to_data):
    labels_directory = path_to_data+"/labels"  # path to your labels directory
    image_directory = path_to_data+"/images"  # path to your images directory
    output_csv_file = "bounding_boxes_with_labels.csv"  # name of the output CSV file

    create_csv(labels_directory, image_directory, output_csv_file)

def prepare_data_to_coco(path_to_data):
    if not os.path.exists("bounding_boxes_with_labels.csv"):
        make_csv(path_to_data)
    df_train = pd.read_csv('bounding_boxes_with_labels.csv')
    x_min = df_train['xmin']
    y_min = df_train['ymin']
    x_max = df_train['xmax']
    y_max = df_train['ymax']
    bboxes = list()
    for i in range(len(x_min)):
        bboxes.append([[x_min[i],y_min[i],x_max[i],y_max[i]]])
    df_train['bboxes'] = bboxes
    df_train["Width"]=416
    df_train["Height"]=416
    train_path = path_to_data+'/images/'
    path = list()
    for i in df_train['image_name']:
        path.append(train_path + i +'.jpg')
    df_train['image_path'] = path
    id = list()
    for i in df_train['image_name']:
        id.append(i[:-4])
    df_train['image_id'] = id
    n_spl=3


    
    gkf  = GroupKFold(n_splits = n_spl) # num_folds=3 as there are total 3 videos
    df_train = df_train.reset_index(drop=True)
    df_train['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, groups = df_train.image_id.tolist())):
        df_train.loc[val_idx, 'fold'] = fold
    # display(df_train.fold.value_counts())
    return df_train

def get_data_dicts(
    _train_df: pd.DataFrame,
    debug: bool = False,
    data_type:str="train"
   
):

    if debug:
        _train_df = _train_df.iloc[:10]  # For debug...
    dataset_dicts = []
    if data_type=="train":
        _train_df=_train_df[_train_df.fold != Selected_Fold]
    else: # val
        _train_df=_train_df[_train_df.fold == Selected_Fold] 
        
    for index, row in tqdm(_train_df.iterrows(), total=len(_train_df)):
        record = {}
        filename  = row.image_path #filename = str(f'{imgdir}/{image_id}.png')
        image_id = row.image_id
        image_height= row.Height
        image_width = row.Width
        bboxes_coco = row.bboxes
        #bboxes_coco  = np.array(row.bboxes).astype(np.float32).copy()
        record["file_name"] = filename
        record["image_id"] = image_id
        record["width"] = image_width
        record["height"] = image_height
        objs = []
        class_id = 0
        for bbox_idx in range(len(bboxes_coco)):
            bbox=bboxes_coco[bbox_idx]
            obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_id,
                }
            objs.append(obj)
            record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def make_coco_dataset(path_to_data):
    df_train=prepare_data_to_coco(path_to_data)
    thing_classes=['car']
    debug=False

    Data_Resister_training="BR_data_train2";
    Data_Resister_valid="BR_data_valid2";


    DatasetCatalog.register(
        Data_Resister_training,
        lambda: get_data_dicts(
            df_train,
            debug=debug,
            data_type="train"
        ),
    )
    MetadataCatalog.get(Data_Resister_training).set(thing_classes=thing_classes)
        

    DatasetCatalog.register(
        Data_Resister_valid,
        lambda: get_data_dicts(
            df_train,
            debug=debug,
            data_type="val"
            ),
        )
    MetadataCatalog.get(Data_Resister_valid).set(thing_classes=thing_classes)
        

    dataset_dicts_train = DatasetCatalog.get(Data_Resister_training)
    metadata_dicts_train = MetadataCatalog.get(Data_Resister_training)

    dataset_dicts_valid = DatasetCatalog.get(Data_Resister_valid)
    metadata_dicts_valid = MetadataCatalog.get(Data_Resister_valid)
    return dataset_dicts_train+dataset_dicts_valid


def get_image_path(image_dir, image_name):
    return os.path.join(image_dir, image_name + ".jpg")

def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
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

def get_accuracy_option2(path_to_data):
    def setup_predictor():
        cfg = get_cfg()
        cfg.merge_from_file("detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = 'cpu'  # Ensure this line is set to 'cpu'
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
        total_false_positive = 0
        total_ground_truth = 0
        
        for image_name, gt_boxes in ground_truth.items():
            image_path = os.path.join(image_dir, image_name + ".jpg")
            predictions = get_predictions(predictor, image_path)
            
            pred_boxes = predictions.pred_boxes.tensor.numpy()
            
            matched_gt_boxes = set()
            matched_pred_boxes = set()
            
            # Check for true positives
            for gt_idx, gt_box in enumerate(gt_boxes):
                gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
                match_found = False
                for pred_idx, pred_box in enumerate(pred_boxes):
                    if check_overlap((gt_xmin, gt_ymin, gt_xmax, gt_ymax), tuple(pred_box)):
                        match_found = True
                        matched_pred_boxes.add(pred_idx)
                        break
                if match_found:
                    matched_gt_boxes.add(gt_idx)
                    total_true_positive += 1
                total_ground_truth += 1
            
            # Check for false positives
            total_false_positive += len(pred_boxes) - len(matched_pred_boxes)
        
        accuracy = total_true_positive / (total_ground_truth + total_false_positive) if total_ground_truth + total_false_positive > 0 else 0
        
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        return accuracy
        #print(f"Accuracy: {accuracy:.2f}")
        #print(f"Time taken: {elapsed_time:.2f} seconds")

    # Usage
    csv_file_path = "bounding_boxes_with_labels.csv"  # path to your CSV file
    image_directory = path_to_data+"/images"  # path to your images directory

    predictor = setup_predictor()

    # Load ground truth and compute accuracy
    ground_truth = load_ground_truth(csv_file_path)
    return compute_accuracy(ground_truth, predictor, image_directory)

def detectron2_evaluation(path_to_data):
    make_coco_dataset(path_to_data)
    return (get_accuracy_option1(path_to_data),get_accuracy_option2(path_to_data))