import grpc
import numpy as np
import cv2
import stereo_vehicle_pb2
import stereo_vehicle_pb2_grpc

def run_find_height(stub, left_image_path, right_image_path, x, y, focal_length, baseline, camera_height):
    # Load and encode the images
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    _, left_image_encoded = cv2.imencode('.png', left_image)
    _, right_image_encoded = cv2.imencode('.png', right_image)

    # Create a HeightRequest
    request = stereo_vehicle_pb2.HeightRequest(
        left_image=left_image_encoded.tobytes(),
        right_image=right_image_encoded.tobytes(),
        x=x,
        y=y,
        focal_length=focal_length,
        baseline=baseline,
        camera_height=camera_height
    )

    # Call the FindHeight method
    response = stub.FindHeight(request)
    print(f"Height at pixel ({x}, {y}): {response.number}")

def run_identify_vehicle(stub, reference_image_path, video_path):
    # Load and encode the reference image
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    _, reference_image_encoded = cv2.imencode('.png', reference_image)

    # Read the video file in binary mode
    with open(video_path, 'rb') as video_file:
        video_data = video_file.read()

    # Create an IdentifyVehicleRequest
    request = stereo_vehicle_pb2.IdentifyVehicleRequest(
        image=reference_image_encoded.tobytes(),
        video=video_data  # Send the binary data of the video
    )

    # Call the IdentifyVehicle method
    response = stub.IdentifyVehicle(request)
    print("Vehicle detection results:", response.results)

def run():
    # Create a channel and a stub
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = stereo_vehicle_pb2_grpc.ImageProcessingStub(channel)

        # Call FindHeight
        run_find_height(stub, 'im0.png', 'im1.png', x=100, y=150, focal_length=800, baseline=0.1, camera_height=1.5)

        # Call IdentifyVehicle
        run_identify_vehicle(stub, 'im0.png', '../video.mp4')

if __name__ == '__main__':
    run()
