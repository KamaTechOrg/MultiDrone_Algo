import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
import cv2
import torch, torchvision
import torchvision.transforms as transforms
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
from detectron2_for_evaluation.config import get_cfg
from detectron2_for_evaluation.engine.defaults import DefaultPredictor
from detectron2_for_evaluation import model_zoo
from utils import initialize_csv, parse_nvidia_smi_output, append_to_csv
from classes_utils import BoundingBox,OutputObject



    





class EvaluationMetrics(metaclass=abc.ABCMeta):
    def __init__(self,predictor:object,images_path:str,labels_path:str) -> None:
        self.__predictor=predictor #model object
        self.__images_path=images_path #path to images dir
        self.__labels_path=labels_path #path to labels dir
        self._monitoring = False #flag to wether gpu is being monitored
        self._stop_event = threading.Event()  # Event to signal stopping of the thread
    
    #predict using self.__predictor the self.__labels_path of the self.__images_path
    @abc.abstractmethod
    def predict(self,one=False, image_path="")->list[OutputObject]:
        pass
        #implement according to you model

    #write gpu utilization to csv
    
    def monitor_gpu(self,csv_file)->None:
        initialize_csv(csv_file)
        while not self._stop_event.is_set():
            # Run the nvidia-smi command
            result = subprocess.run(['nvidia-smi', '-q', '-i', '0', '-d', 'UTILIZATION'], capture_output=True, text=True)
            output = result.stdout

            # Parse the output
            data = parse_nvidia_smi_output(output)

            # Append the data to CSV
            append_to_csv(data,csv_file)

            # Wait before the next read
            time.sleep(1)

    #evaluate gpu utilization
    def gpu_evaluation(self, csv_file_path: str) -> None:
        self._monitoring = True
        self._stop_event.clear()  # Clear the stop event before starting the thread
        self.monitor_thread = threading.Thread(target=self.monitor_gpu, args=(csv_file_path,))
        self.monitor_thread.start()

        self.predict()

        self._monitoring = False
        self._stop_event.set()  # Signal the thread to stop
        self.monitor_thread.join()  # Wait for the thread to finish

    def stop_monitoring(self):
        self._stop_event.set()  # Manually signal the thread to stop
        self.monitor_thread.join()  # Wait for the thread to finish

    #evaluate time for predictions
    def time_evaluation(self)->float:
        start=time.time()
        self.predict()
        end=time.time()
        return end-start
    


    #needs to be decided which metrics to implement
    
    #precision
    def metric_1(self,csv_file)->float:

        total_true_positive = 0
        total_ground_truth = 0

        ground_truth=self.load_ground_truth(csv_file)
        for image_name, gt_boxes in list(ground_truth.items())[:100]:
            # image_path = os.path.join(self.__images_path)
            prediction = self.predict(one=True, image_path=image_name)[0]
            
            pred_boxes = prediction.bboxes
            
            # Check if any predictions overlap with ground truth
            for gt_box in gt_boxes:
                gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
                match_found = False
                for pred_box in pred_boxes:
                    if self.check_overlap((gt_xmin, gt_ymin, gt_xmax, gt_ymax), tuple(pred_box.get_bbox())):
                        match_found = True
                        break
                if match_found:
                    total_true_positive += 1
                total_ground_truth += 1
        
        accuracy = total_true_positive / total_ground_truth if total_ground_truth > 0 else 0
    
        return accuracy
    
    def check_overlap(self,box1, box2):
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        # Check if there is any overlap
        if x1 < x2_p and x2 > x1_p and y1 < y2_p and y2 > y1_p:
            return True
        return False
    
    #true classes
    def load_ground_truth(self,csv_file):
        df = pd.read_csv(csv_file)
        ground_truth = {}
        for _, row in df.iterrows():
            # Extract the necessary information from the row
            image_name = row["image_path"]
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])

            # Add the bounding box information to the ground_truth dictionary
            if image_name not in ground_truth:
                ground_truth[image_name] = []
            ground_truth[image_name].append((xmin, ymin, xmax, ymax))
        return ground_truth 

    def metric_2(self):
        pass

    def metric_3(self):
        pass


