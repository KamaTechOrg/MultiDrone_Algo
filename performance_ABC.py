from abc import ABC, abstractmethod
import torch
import time

class CheckPerformance(ABC):
    @staticmethod
    @abstractmethod
    def measure_inference_time(model, data, device):
        pass

    @staticmethod
    @abstractmethod
    def check_gpu():
        pass
 
    @staticmethod
    @abstractmethod
    def check_gpu_performance_and_memory(model, data):
        pass
    