import cv2
import numpy as np
import os
import tempfile
def is_car_in_frame(car_image, frame, sift, bf, threshold=10):
    kp1, des1 = sift.detectAndCompute(car_image, None)
    kp2, des2 = sift.detectAndCompute(frame, None)
    
    if des1 is None or des2 is None:
        return False, None, None, None

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return len(good_matches) >= threshold, kp1, kp2, good_matches

def process_video_from_memory(video_data, car_image, output_folder, threshold=10, duration_sec=30):
    car_image_color = cv2.cvtColor(car_image, cv2.COLOR_GRAY2BGR)
    print("Processing video...")

    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_data)
        temp_video_path = temp_video_file.name

    # Open the video file using VideoCapture
    cap = cv2.VideoCapture(temp_video_path)

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    bool_array = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    max_frames = duration_sec * fps
    frame_count = 0

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check if the vehicle is in the frame
        is_detected, kp1, kp2, good_matches = is_car_in_frame(car_image, gray_frame, sift, bf, threshold)
        bool_array.append(is_detected)

        # Show matches
        frame_color = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        frame_with_matches = cv2.drawMatches(car_image_color, kp1, frame_color, kp2, good_matches, None,
                                             matchColor=(0, 255, 0), singlePointColor=None,
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Add notification if vehicle is detected
        if is_detected:
            cv2.putText(frame_with_matches, 'Car Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save the result
        result_text = 'True' if is_detected else 'False'
        output_filename = os.path.join(output_folder, f"frame_{frame_count + 1}_{result_text}.png")
        cv2.imwrite(output_filename, frame_with_matches)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Clean up temporary video file
    os.remove(temp_video_path)

    return bool_array
