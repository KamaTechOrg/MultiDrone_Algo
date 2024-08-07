import GPUtil
import time
import torch
from performance_ABC import CheckPerformance

class PerformanceChecker(CheckPerformance):
    @staticmethod
    def check_gpu_performance_and_memory(model, data, device):
        model.eval()
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        gpu_memory_before = torch.cuda.memory_allocated()
        with torch.no_grad():
            model.to(device)
            for img_pair in data:
                img1, img2 =  img_pair[0].cuda(), img_pair[1].cuda()
                _ = model({"image0": img1, "image1": img2})
        gpu_time = time.time() - start_time
        gpu_memory_after = torch.cuda.memory_allocated()
        gpu_memory = torch.cuda.memory_allocated() 
        peak_memory = torch.cuda.max_memory_allocated()
        gpu_memory_usage = gpu_memory_after - gpu_memory_before
        model.cpu()
        return {
            'GPU Time (s)': gpu_time,
            'GPU Memory Usage (MB)': gpu_memory/ (1024 ** 2),
            'Peak GPU Memory Usage (MB)': peak_memory / 1024 / 1024
        }
          
