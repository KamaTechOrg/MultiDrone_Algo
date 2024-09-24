import grpc
import stereo_vehicle_pb2
import stereo_vehicle_pb2_grpc
import cv2
import time

# Helper function to read an image and convert it to bytes
def read_image_as_bytes(image_path):
    with open(image_path, 'rb') as img_file:
        return img_file.read()

# Helper function to read a video and convert it to bytes
def read_video_as_bytes(video_path):
    with open(video_path, 'rb') as video_file:
        return video_file.read()


def test_control_thread(stub):
    # Load video data
    video_bytes = read_video_as_bytes('C:\BootCamp\project\pythonProject\\video stabilization\input2.mp4')

    # Activate the thread with video data
    response = stub.ControlThread(stereo_vehicle_pb2.ControlRequest(activate=True, video=video_bytes))
    print(f"ControlThread activate response: {response.message}")

    # Let the background process run for a few seconds
    time.sleep(30)

    # Deactivate the thread
    response = stub.ControlThread(stereo_vehicle_pb2.ControlRequest(activate=False))
    print(f"ControlThread deactivate response: {response.message}")

def run_all_tests():
    # Connect to the gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    stub = stereo_vehicle_pb2_grpc.ImageProcessingStub(channel)


    # Test the ControlThread RPC
    print("Testing ControlThread...")
    test_control_thread(stub)

if __name__ == '__main__':
    run_all_tests()
