import grpc
from concurrent import futures
import cv2
import numpy as np
import stereo_vehicle_pb2
import stereo_vehicle_pb2_grpc
import time
import os
import threading
import logic_functions.find_height as calc_height
import logic_functions.stabilization as calc_stabilization
import logic_functions.detect_new as calc_detect
# Global flag and video data for controlling the background thread
activate_thread = True
video_data = None  # Will hold video data when activated

# Background task that will run when activated
saved_video_path = 'input2.mp4'


# Background task that will run when activated
def background_task():
    global video_data
    while True:
        if activate_thread:
            if video_data:
                print("Saving video data to file...")

                # Save the video to a file
                with open(saved_video_path, 'wb') as video_file:
                    video_file.write(video_data)

                print(f"Video saved to {saved_video_path}")

                # After saving the video, read and process it inside stabilize_frames_sift_optical_flow
                stabilized_frames = calc_stabilization.stabilize_video_sift_optical_flow(saved_video_path)

                # e.g., perform further operations on stabilized_frames
                print("Video stabilization and processing completed.")
                video_data = None  # Clear video data after processing
            time.sleep(1)
        else:
            time.sleep(1)  # Sleep when not active


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

        # Placeholder logic for height calculation
        height = calc_height.calculate_height_above_ground(x, y, left_image, right_image, focal_length, baseline,
                                                           camera_height)

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

        my_video_data = request.video  # For simplicity, assume it's binary data for now
        img_path = 'detect_image.png'
        video_path = 'detect_video.mp4'
        try:
            with open(img_path, 'wb') as image_file:
                image_file.write(reference_image_data)
                print(f"Reference image saved to {img_path}")

            # Save the video to a file
            with open(video_path, 'wb') as video_file:
                video_file.write(my_video_data)
                print(f"Video saved to {video_path}")

            print(f"Received reference image and video for vehicle identification")
            output_folder = 'output_frames'
            # Placeholder logic to detect the vehicle in frames
            results = calc_detect.process_video(video_path, img_path, output_folder)

            # Return boolean array with detection results
            return stereo_vehicle_pb2.BooleanArrayResponse(results=results)
        finally:
        # Ensure files are deleted after processing, regardless of success or error
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Deleted {img_path} from the server.")

            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted {video_path} from the server.")


    # Method 3: Control the background thread (new)
    def ControlThread(self, request, context):
        global activate_thread, video_data
        if request.activate:
            activate_thread = True
            video_data = request.video  # Store video data if present
            return stereo_vehicle_pb2.ControlResponse(message="Thread activated and video received.")
        else:
            activate_thread = False
            video_data = None  # Clear video data
            return stereo_vehicle_pb2.ControlResponse(message="Thread deactivated.")


#######################
    def GetStabilizedFrame(self, request, context):
        frame = calc_stabilization.get_stabilized_frame()
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            return stereo_vehicle_pb2.FrameResponse(frame=buffer.tobytes())
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('No stabilized frame available')
            return stereo_vehicle_pb2.FrameResponse()
#######################

# Start the gRPC server
def serve():
    # Start the background task thread
    thread = threading.Thread(target=background_task)
    thread.daemon = True  # Daemon thread to exit when main program exits
    thread.start()

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
