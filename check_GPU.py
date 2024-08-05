
import subprocess
import time
import csv
import re
import torch
from ultralytics import YOLO

# File to write the CSV data
csv_file = 'gpu_utilization.csv'

# Function to parse the nvidia-smi output and extract relevant data
def parse_nvidia_smi_output(output):
    data = {
        'timestamp': '',
        'gpu_utilization': '',
        'memory_utilization': '',
        'encoder_utilization': '',
        'decoder_utilization': ''
    }
    # Regular expressions for extracting data
    gpu_pattern = re.compile(r"Gpu\s+:\s+(\d+)\s*%")
    memory_pattern = re.compile(r"Memory\s+:\s+(\d+)\s*%")
    encoder_pattern = re.compile(r"Encoder\s+:\s+(\d+)\s*%")
    decoder_pattern = re.compile(r"Decoder\s+:\s+(\d+)\s*%")
    
    # Extract utilization data
    gpu_match = gpu_pattern.search(output)
    if gpu_match:
        data['gpu_utilization'] = gpu_match.group(1)
    memory_match = memory_pattern.search(output)
    if memory_match:
        data['memory_utilization'] = memory_match.group(1)
    encoder_match = encoder_pattern.search(output)
    if encoder_match:
        data['encoder_utilization'] = encoder_match.group(1)
    decoder_match = decoder_pattern.search(output)
    if decoder_match:
        data['decoder_utilization'] = decoder_match.group(1)
        
    data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")  # Add current timestamp
    
    return data

# Initialize CSV file and write header if it does not exist
def initialize_csv():
    try:
        with open(csv_file, 'x', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['timestamp', 'gpu_utilization', 'memory_utilization', 'encoder_utilization', 'decoder_utilization'])
            writer.writeheader()
    except FileExistsError:
        pass

# Append data to the CSV file
def append_to_csv(data):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['timestamp', 'gpu_utilization', 'memory_utilization', 'encoder_utilization', 'decoder_utilization'])
        writer.writerow(data)

def setup_yolo_model(model_path):
    model = YOLO(model_path)
    return model

def run_inference(model, image_path):
    results = model(image_path)
    return results

# Main function to run the script
def main(model_path, image_path):
    initialize_csv()
    model = setup_yolo_model(model_path)
    
    while True:
        # Run inference with YOLO
        results = run_inference(model, image_path)
        
        # Run the nvidia-smi command
        result = subprocess.run(['nvidia-smi', '-q', '-d', 'UTILIZATION'], capture_output=True, text=True)
        output = result.stdout
        
        # Parse the output
        data = parse_nvidia_smi_output(output)
        
        # Append the data to CSV
        append_to_csv(data)
        
        # Wait before the next read
        time.sleep(1)  # Increased sleep time to avoid excessive logging

if __name__ == "__main__":
    # Replace 'yolov8n.pt' and 'image.jpg' with your actual model and image paths
    main('yolov8n.pt', 'bus.jpg')
