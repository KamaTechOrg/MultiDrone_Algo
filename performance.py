import GPUtil
import time
import torch
from performance_ABC import CheckPerformance

class PerformanceChecker(CheckPerformance):
    @staticmethod
    def check_gpu_performance_and_memory(model, data):
        model.eval()
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        gpu_memory_before = torch.cuda.memory_allocated()
        with torch.no_grad():
            model.cuda()
            for img_pair in data:
                img1, img2 = img_pair[0].cuda(), img_pair[1].cuda()
                _ = model({"image0": img1, "image1": img2})
        gpu_time = time.time() - start_time
        gpu_memory_after = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        gpu_memory_usage = gpu_memory_after - gpu_memory_before
        model.cpu()
        return {
            'GPU Time (s)': gpu_time,
            'GPU Memory Usage (MB)': gpu_memory_usage / 1024 / 1024,
            'Peak GPU Memory Usage (MB)': peak_memory / 1024 / 1024
        }
    
    @staticmethod
    def measure_inference_time(model, data, device):
        model.to(device)
        data = {k: v.to(device) for k, v in data.items()}
        # Warm-up
        for _ in range(10):
            _ = model(data)
        with torch.no_grad():
            start_time = time.time()
            for _ in range(100):
                _ = model(data)
            end_time = time.time()
        avg_time = (end_time - start_time) / 100
        return f"Average inference time: {avg_time:.6f} seconds"

    @staticmethod
    def check_gpu():
        if torch.cuda.is_available():
            gpu_info = [
                "CUDA is available. Using GPU.",
                f"Number of GPUs: {torch.cuda.device_count()}"
            ]
            for i in range(torch.cuda.device_count()):
                gpu_info.extend([
                    f"GPU {i}: {torch.cuda.get_device_name(i)}",
                    f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB",
                    f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.1f} GB"
                ])
            return "\n".join(gpu_info)
        else:
            return "CUDA is not available. Using CPU."
