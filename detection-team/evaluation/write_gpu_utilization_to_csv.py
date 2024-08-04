import subprocess
import time
import csv
import re

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
    timestamp_pattern = re.compile(r"Timestamp\s+:\s+(.*)")
    gpu_pattern = re.compile(r"Gpu\s+:\s+(\d+)\s*%")
    memory_pattern = re.compile(r"Memory\s+:\s+(\d+)\s*%")
    encoder_pattern = re.compile(r"Encoder\s+:\s+(\d+)\s*%")
    decoder_pattern = re.compile(r"Decoder\s+:\s+(\d+)\s*%")

    # Extract the timestamp
    timestamp_match = timestamp_pattern.search(output)
    if timestamp_match:
        data['timestamp'] = timestamp_match.group(1)

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

# Main function to run the script
def main():
    initialize_csv()
    while True:
        # Run the nvidia-smi command
        result = subprocess.run(['nvidia-smi', '-q', '-i', '0', '-d', 'UTILIZATION'], capture_output=True, text=True)
        output = result.stdout

        # Parse the output
        data = parse_nvidia_smi_output(output)

        # Append the data to CSV
        append_to_csv(data)

        # Wait before the next read
        time.sleep(1)

if __name__ == "__main__":
    main()
