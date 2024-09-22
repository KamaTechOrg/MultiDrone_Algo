import cv2
import numpy as np
# פרמטרים של המצלמה
focal_length = 1733.74  # אורך מוקד
baseline = 536.62       # המרחק בין המצלמות
camera_height = 100     # גובה המצלמה מעל פני הקרקע
def calculate_height_above_ground(x, y, image_left, image_right,focal_length,baseline,camera_height):
    # image_left = cv2.imread(image_left_path, cv2.IMREAD_GRAYSCALE)
    # image_right = cv2.imread(image_right_path, cv2.IMREAD_GRAYSCALE)
    if image_left is None or image_right is None:
        raise ValueError("לא ניתן לקרוא את אחת מהתמונות")
    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=80,
        blockSize=5
    )
    disparity = stereo_sgbm.compute(image_left, image_right)
    disparity = np.array(disparity, dtype=np.float32) / 16.0
    if 0 <= y < disparity.shape[0] and 0 <= x < disparity.shape[1]:
        disparity_value = disparity[y, x]
        if disparity_value > 0:
            depth = (baseline * focal_length) / disparity_value
            height_above_ground = camera_height - depth
            return height_above_ground
        else:
            return None
    else:
        return None
    
# קואורדינטות הפיקסל בתמונה
x = 500
y = 600
# חישוב והדפסת הגובה מעל פני הקרקע
# height_above_ground = calculate_height_above_ground(x, y, "im0.png", "im1.png")
# if height_above_ground is not None:
#     print(f"הגובה של הפיקסל ({x}, {y}) מעל פני הקרקע הוא: {height_above_ground:.2f} מטרים")
# else:
#     print("הקואורדינטות מחוץ לתחום התמונה.")