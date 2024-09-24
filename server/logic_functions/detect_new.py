#best till here with registration show, works really good
#Needs improvements with far car which is too small to recognize and also doesnt have all the 3d elements
import cv2
import os
import numpy as np

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

def process_video(video_path, car_image_path, output_folder, threshold=10, duration_sec=30):
    car_image = cv2.imread(car_image_path, cv2.IMREAD_GRAYSCALE)
    car_image_color = cv2.imread(car_image_path)  # קריאה בצבע לצורך הצגה עם קווים
    
    cap = cv2.VideoCapture(video_path)
    
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    bool_array = []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    max_frames = duration_sec * fps  # מספר הפריימים שנרצה לעבור עליהם (חצי דקה)
    frame_count = 0
    
    # יצירת תיקיית הפלט אם אינה קיימת
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # בדיקה אם הרכב נמצא בפריים
        is_detected, kp1, kp2, good_matches = is_car_in_frame(car_image, gray_frame, sift, bf, threshold)
        bool_array.append(is_detected)

        # הצגת הקווים של ההתאמות בין התמונות
        frame_color = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # המרה לצבע לצורך הצגת קווים
        frame_with_matches = cv2.drawMatches(car_image_color, kp1, frame_color, kp2, good_matches, None, 
                                             matchColor=(0, 255, 0), singlePointColor=None, 
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # הוספת התראה על המסך אם הרכב זוהה
        if is_detected:
            cv2.putText(frame_with_matches, 'Car Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # שמירת התמונה עם התוצאה
        result_text = 'True' if is_detected else 'False'
        output_filename = os.path.join(output_folder, f"frame_{frame_count + 1}_{result_text}.png")
        cv2.imwrite(output_filename, frame_with_matches)

        # הצגת הווידאו
        # cv2.imshow('Video', frame_with_matches)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # יציאה מהווידאו בלחיצה על Q
        #     break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    return bool_array

# שימוש בקוד
video_path = 'rentis_road.mp4'
car_image_path = 'red_track.png'
output_folder = 'output_frames'
result = process_video(video_path, car_image_path, output_folder, threshold=10, duration_sec=30)

print(result)