
import cv2

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
import numpy as np
# import List
import seaborn as sns
from utils import initialize_csv, parse_nvidia_smi_output, append_to_csv
from classes_utils import BoundingBox,OutputObject



    


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
    def metric_1(self,confusion_matrix:np.ndarray)->float:

        cm = confusion_matrix
        true_positives = cm[0, 0]
        # false_negatives = cm[0, 1]
        false_positives = cm[1, 0]
        
        # Calculate precision
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
        return precision
    
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
            category_id=int(row["category_id"])

            # Add the bounding box information to the ground_truth dictionary
            if image_name not in ground_truth:
                ground_truth[image_name] = []
            ground_truth[image_name].append((xmin, ymin, xmax, ymax,category_id))
        return ground_truth 

   

    def compute_confusion_matrix(self, ground_truth_csv: str) -> np.ndarray:
        ground_truth = self.load_ground_truth(ground_truth_csv)
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        
        for image_name, gt_boxes in list(ground_truth.items())[:10]:
            predictions = self.predict(one=True, image_path=image_name)[0]
            pred_boxes = predictions.bboxes
            
            matched_gt_indices = set()
            matched_pred_indices = set()
            
            for i, gt_box in enumerate(gt_boxes):
                gt_bbox = gt_box[:4]  # Extract (xmin, ymin, xmax, ymax)
                
                for j, pred_box in enumerate(pred_boxes):
                    pred_bbox = pred_box.get_bbox()
                    
                    if self.check_overlap(gt_bbox, pred_bbox):
                        matched_gt_indices.add(i)
                        matched_pred_indices.add(j)
                        true_positives += 1
                        break
            
            # Count false negatives (ground truth boxes not matched)
            false_negatives += len(gt_boxes) - len(matched_gt_indices)
            
            # Count false positives (predicted boxes not matched)
            false_positives += len(pred_boxes) - len(matched_pred_indices)
        
        # Build the confusion matrix
        cm = np.array([
            [true_positives, false_negatives],  # Row for True Positives and False Negatives
            [false_positives, 0]  # Row for False Positives (no True Negatives in this context)
        ])
        # print(f'true_positive: {true_positives},\nfalse_negative: {false_negatives},\nfalse_positive: {false_positives}')
        return cm

    
    def get_class_from_box(self, box:BoundingBox):
        # print(box)
    # If class information is stored separately, retrieve the class from a mapping or list
        class_index = box.get_category()-1  # or box[4] if class is at index 4 in the list/tuple
        class_names = self.get_class_names()  # Ensure this method returns the correct class names list
        return class_names[class_index] if class_index < len(class_names) else None
    
    def get_class_names(self) -> list[str]:
        # ground_truth_csv = self.__labels_path
        # _, class_names = self.compute_confusion_matrix(ground_truth_csv)
        return ['car', 'truck', 'bus']
    
    def display_confusion_matrix(self, confusion_matrix:np.ndarray):
        # print("hello")
        # confusion_matrix=self.compute_confusion_matrix(csv_file)
        # class_names=self.get_class_names()
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['positive','negative'], yticklabels=['true','false'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def compute_recall(self, confusion_matrix:np.ndarray) -> np.ndarray:
        # confusion_matrix=self.compute_confusion_matrix(csv_file)
        true_positives = confusion_matrix[0,0]
        false_negatives = confusion_matrix[0,1]
        recall = np.divide(true_positives, true_positives + false_negatives, out=np.zeros_like(true_positives, dtype=float), where=(true_positives + false_negatives) != 0)
        return recall
    
    def recall(self, confusion_matrix:np.ndarray) -> np.ndarray:
        # cm, _ = self.compute_confusion_matrix(csv_file)
        recall_scores = self.compute_recall(confusion_matrix)
        return recall_scores
    
    def metric_2(self, confusion_matrix:np.ndarray):
        # cm, classes = self.compute_confusion_matrix(ground_truth_csv)
        # self.display_confusion_matrix(confusion_matrix)
        recall_scores = self.compute_recall(confusion_matrix)
        print(f"Recall: {recall_scores}")
        # print("Recall scores for each class:")
        # for class_name, recall in zip(classes, recall_scores):
        #     print(f"Class {class_name}: Recall = {recall:.4f}")

    def metric_3(self, confusion_matrix:np.ndarray)->None:
        precision=self.metric_1(confusion_matrix)
        recall=self.recall(confusion_matrix)
        self.display_confusion_matrix(confusion_matrix)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"F1: {f1}")


