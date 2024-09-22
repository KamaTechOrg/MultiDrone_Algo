import grpc
from concurrent import futures
import cv2
import numpy as np
import stereo_vehicle_pb2
import stereo_vehicle_pb2_grpc
import time
import find_height
import detect_car_save
# Implement the ImageProcessing service
class ImageProcessingServicer(stereo_vehicle_pb2_grpc.ImageProcessingServicer):
    
    # Method 1: Find the height of a pixel
    def FindHeight(self, request, context):
        # Decode images from binary data
        left_image_data = np.frombuffer(request.left_image, np.uint8)
        right_image_data = np.frombuffer(request.right_image, np.uint8)
        
        left_image = cv2.imdecode(left_image_data, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imdecode(right_image_data, cv2.IMREAD_GRAYSCALE)

        # Extract camera parameters and coordinates
        x = request.x
        y = request.y
        focal_length = request.focal_length
        baseline = request.baseline
        camera_height = request.camera_height

        print(f"Calculating height for pixel ({x}, {y}) with camera parameters")

        # Calculate the height above the ground
        height = find_height.calculate_height_above_ground(x, y, left_image, right_image, focal_length, baseline, camera_height)

        if height is not None:
            return stereo_vehicle_pb2.NumberResponse(number=height)
        else:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('Invalid coordinates or no valid depth found')
            return stereo_vehicle_pb2.NumberResponse(number=-1)

    # Method 2: Identify vehicle in the video
    def IdentifyVehicle(self, request, context):
        # Decode reference image and video data
        reference_image_data = np.frombuffer(request.image, np.uint8)
        reference_image = cv2.imdecode(reference_image_data, cv2.IMREAD_GRAYSCALE)

        video_data = request.video  # For simplicity, assume it's binary data for now

        print(f"Received reference image and video for vehicle identification")
        output_folder = 'output_frames'
        # Placeholder logic to detect the vehicle in frames
        results = detect_car_save.process_video_from_memory(video_data, reference_image,output_folder)

        # Return boolean array with detection results
        return stereo_vehicle_pb2.BooleanArrayResponse(results=results)

# Start the gRPC server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_send_message_length', 100 * 1024 * 1024)  # 100 MB
        ])
    stereo_vehicle_pb2_grpc.add_ImageProcessingServicer_to_server(ImageProcessingServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051...")
    try:
        while True:
            time.sleep(86400)  # Run server for 24 hours
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
