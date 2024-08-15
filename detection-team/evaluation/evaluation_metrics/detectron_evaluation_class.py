# import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
# from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.structures import BoxMode
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
# from detectron2.data import build_detection_test_loader
# from detectron2.utils.visualizer import ColorMode
import cv2
# import torch, torchvision
# import torchvision.transforms as transforms
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import subprocess
import time
import csv
import re
import abc
import time
import threading
import subprocess
import pandas as pd
import os
import os
import cv2
import torch
import pandas as pd
import time
import sys
import importlib
from classes_utils import BoundingBox,OutputObject
from detectron2_for_evaluation.config import get_cfg
from detectron2_for_evaluation.engine.defaults import DefaultPredictor
from detectron2_for_evaluation import model_zoo
from abstract_class import EvaluationMetrics

# Append data to the CSV file
def append_to_csv(data,csv_file):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['timestamp', 'gpu_utilization', 'memory_utilization', 'encoder_utilization', 'decoder_utilization'])
        writer.writerow(data)


def convert_detectron2_output_to_output_object(image_path: str, detectron2_output) -> OutputObject:
        """
        Converts Detectron2 output to OutputObject.

        Args:
        - image_path (str): The path to the image.
        - detectron2_output: The output from Detectron2 model, typically a dictionary 
        with 'instances' containing bounding boxes, category_ids, etc.

        Returns:
        - OutputObject: The converted object containing the bounding boxes and category ID.
        """
        # Assuming detectron2_output['instances'] is a Instances object
        instances = detectron2_output['instances']
        
        # Get predicted bounding boxes (Tensor of shape [N, 4])
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        
        # Get predicted category IDs (Tensor of shape [N])
        category_ids = instances.pred_classes.cpu().numpy()

        # Convert bounding boxes to BoundingBox instances
        bboxes = [BoundingBox(category_id, xmin, ymin, xmax, ymax) for (xmin, ymin, xmax, ymax) , category_id in zip(boxes,category_ids)]

        # Create and return the OutputObject
        return OutputObject(image_path, bboxes)






class DetectronEvluation(EvaluationMetrics):
    
    def __init__(self,images_path: str) -> None:
        predictor=self.setup_predictor()
        super().__init__(predictor, images_path, "")
    
    


    def predict(self,one=False, image_path="") -> list:
        if one:
            return [self.get_predictions(self._EvaluationMetrics__predictor,image_path)]
        
        return (self.recursive_predict(self._EvaluationMetrics__images_path,10))[0]
    
    def recursive_predict(self, images_dir:str,count:int)->list:
        outputs=[]
        
        for img in os.listdir(images_dir):
            if count<=0:
                break
            diff=1
            # print(img)
            # print(os.path.isdir(os.path.join(images_dir,img)))
            if os.path.isdir(os.path.join(images_dir,img)):
                output,diff=self.recursive_predict(os.path.join(images_dir,img),count)
                outputs+=output
                diff=10-diff
            else:
                # print("hello",count)
                # if os.path.isfile(img) and (os.path.splitext(os.path.basename(img)))[-1] in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}:
                outputs+=[self.get_predictions(self._EvaluationMetrics__predictor,os.path.join(images_dir,img))]
            count-=diff
        return outputs,count

    
     
    def setup_predictor(self):
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

    def get_predictions(self,predictor, image_path):
        im = cv2.imread(image_path)
        print(image_path)
        outputs = predictor(im)
       

        return convert_detectron2_output_to_output_object(image_path,outputs)  
    # def compute_confusion_matrix(self, ground_truth_csv: str) -> np.ndarray:
    #     return super().compute_confusion_matrix(ground_truth_csv)
    # def display_confusion_matrix(self, csv_file: str):
    #     return super().display_confusion_matrix(csv_file)
    
    # def compute_recall(self, csv_file:str) -> np.ndarray:
    #     confusion_matrix=self.compute_confusion_matrix(csv_file)
    #     return super().compute_recall(confusion_matrix)

    



    


