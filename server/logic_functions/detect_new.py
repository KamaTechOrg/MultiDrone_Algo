import cv2
import os
import numpy as np

# Function to check if the car is present in the current frame
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

# Function to process the video and check each frame for the presence of the car
def process_video(video_path, car_image_path, output_folder, threshold=10, duration_sec=30):
    car_image = cv2.imread(car_image_path, cv2.IMREAD_GRAYSCALE)
    car_image_color = cv2.imread(car_image_path)  # Read in color for displaying lines
    
    cap = cv2.VideoCapture(video_path)
    
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    bool_array = []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    max_frames = duration_sec * fps  # Total frames to process (30 seconds)
    frame_count = 0
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if the car is detected in the current frame
        is_detected, kp1, kp2, good_matches = is_car_in_frame(car_image, gray_frame, sift, bf, threshold)
        bool_array.append(is_detected)

        # Display matching lines between images
        frame_color = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Convert to color for displaying lines
        frame_with_matches = cv2.drawMatches(car_image_color, kp1, frame_color, kp2, good_matches, None, 
                                             matchColor=(0, 255, 0), singlePointColor=None, 
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Add an alert on the screen if the car is detected
        if is_detected:
            cv2.putText(frame_with_matches, 'Car Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save the result image with detection info
        result_text = 'True' if is_detected else 'False'
        output_filename = os.path.join(output_folder, f"frame_{frame_count + 1}_{result_text}.png")
        cv2.imwrite(output_filename, frame_with_matches)

        # Uncomment this section to display the video in real-time
        # cv2.imshow('Video', frame_with_matches)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'Q' to exit the video
        #     break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    return bool_array

# Usage of the code
video_path = 'rentis_road.mp4'
car_image_path = 'red_track.png'
output_folder = 'output_frames'
result = process_video(video_path, car_image_path, output_folder, threshold=10, duration_sec=30)

print(result)
