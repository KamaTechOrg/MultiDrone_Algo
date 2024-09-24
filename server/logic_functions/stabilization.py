import cv2
import numpy as np
import queue

frame_queue = queue.Queue()

def weighted_moving_average(curve, radius):
    weights = np.linspace(1, 0, 2 * radius + 1)
    weights /= np.sum(weights)
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, weights, mode='same')
    return curve_smoothed[radius:-radius]

def fix_border(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.05)
    return cv2.warpAffine(frame, T, (s[1], s[0]))

def stabilize_video_sift_optical_flow(input_path, smoothing_radius=300, motion_threshold=0.15):
    cap = cv2.VideoCapture(input_path)
    sift = cv2.SIFT_create()
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = sift.detectAndCompute(prev_gray, None)
    prev_pts = np.array([kp.pt for kp in kp_prev], dtype=np.float32).reshape(-1, 1, 2)

    transforms = []
    trajectory = []
    last_stabilized_frame = prev_frame.copy()
    smoothed_trajectory = np.zeros((3,), np.float32)  # Initialize smoothed_trajectory

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_next, des_next = sift.detectAndCompute(gray, None)
        next_pts = np.array([kp.pt for kp in kp_next], dtype=np.float32).reshape(-1, 1, 2)
        next_pts_opt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, maxLevel=4)

        if next_pts_opt is None or status is None:
            stabilized_frame = last_stabilized_frame
            frame_queue.put(stabilized_frame)
            continue

        good_prev_pts = prev_pts[status == 1]
        good_next_pts = next_pts_opt[status == 1]

        if len(good_prev_pts) >= 4 and len(good_next_pts) >= 4:
            H, inliers = cv2.estimateAffinePartial2D(good_prev_pts, good_next_pts, method=cv2.RANSAC,
                                                     ransacReprojThreshold=3)

            if H is not None:
                dx = H[0, 2]
                dy = H[1, 2]
                da = np.arctan2(H[1, 0], H[0, 0])

                if np.abs(dx) < motion_threshold:
                    dx = 0
                if np.abs(dy) < motion_threshold:
                    dy = 0
                if np.abs(da) < motion_threshold:
                    da = 0

                transform = np.array([dx, dy, da])
                transforms.append(transform)
                trajectory.append(transform)

                if len(transforms) > smoothing_radius:
                    smoothed_trajectory = np.mean(transforms[-smoothing_radius:], axis=0)

                difference = smoothed_trajectory - np.sum(trajectory, axis=0)
                dx += difference[0]
                dy += difference[1]
                da += difference[2]

                H_corrected = np.array([
                    [np.cos(da), -np.sin(da), dx],
                    [np.sin(da), np.cos(da), dy]
                ], dtype=np.float32)

                stabilized_frame = cv2.warpAffine(frame, H_corrected, (frame.shape[1], frame.shape[0]))
                stabilized_frame = fix_border(stabilized_frame)
                last_stabilized_frame = stabilized_frame
            else:
                stabilized_frame = frame
        else:
            stabilized_frame = last_stabilized_frame
        
        frame_queue.put(stabilized_frame)
        prev_gray = gray
        prev_pts = good_next_pts.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()

def get_stabilized_frame():
    try:
        if not frame_queue.empty():
            return frame_queue.get()
        else:
            print("Queue is empty.")
            return None
    except Exception as e:
        print(f"Error retrieving frame: {e}")
        return None

if __name__ == '__main__':
    stabilize_video_sift_optical_flow('C:\\BootCamp\\project\\pythonProject\\video stabilization\\input2.mp4', smoothing_radius=300, motion_threshold=0.15)
