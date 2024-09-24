import grpc
import connection_to_detection.server.stereo_vehicle_pb2 as stereo_vehicle_pb2
import connection_to_detection.server.stereo_vehicle_pb2_grpc as stereo_vehicle_pb2_grpc
import cv2
import numpy as np
import time


def fetch_stabilized_frames():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = stereo_vehicle_pb2_grpc.ImageProcessingStub(channel)

        while True:
            try:
                frame_request = stereo_vehicle_pb2.Empty()
                response = stub.GetStabilizedFrame(frame_request)

                if response.frame:
                    nparr = np.frombuffer(response.frame, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    cv2.imshow('Stabilized Frame', frame)
                    cv2.waitKey(1)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("No stabilized frame available yet.")

                time.sleep(1)  # המתנה קצרה לפני קריאה נוספת

            except grpc.RpcError as e:
                print(f"gRPC error: {e.code()} - {e.details()}")
                break


if __name__ == '__main__':
    fetch_stabilized_frames()
