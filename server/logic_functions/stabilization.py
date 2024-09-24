import cv2
import numpy as np
import queue
import os

frame_queue = queue.Queue()


def save_frame_to_folder(frame, folder_path, frame_number):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    frame_filename = os.path.join(folder_path, f"frame_{frame_number:04d}.png")
    cv2.imwrite(frame_filename, frame)


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


def detect_static_objects(prev_pts, curr_pts, camera_transform, threshold=2.0):
    dx_camera, dy_camera, da_camera = camera_transform
    object_displacements = np.linalg.norm(curr_pts - prev_pts, axis=1)
    camera_displacement = np.linalg.norm([dx_camera, dy_camera])

    static_points_prev = []
    static_points_curr = []

    for i, displacement in enumerate(object_displacements):
        if abs(displacement) <= abs(camera_displacement):
            static_points_prev.append(prev_pts[i])
            static_points_curr.append(curr_pts[i])
    return np.array(static_points_prev), np.array(static_points_curr)


def stabilize_video_sift_optical_flow(input_path, output_folder='stabilized_frames', smoothing_radius=50,
                                      motion_threshold=0.15):
    cap = cv2.VideoCapture(input_path)
    frame_number = 0

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create SIFT detector and BFMatcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = sift.detectAndCompute(prev_gray, None)

    transforms = []
    trajectory = []
    smoothed_trajectory = np.zeros((3,), np.float32)
    last_stabilized_frame = prev_frame.copy()

    while True:
        print(1)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_next, des_next = sift.detectAndCompute(gray, None)

        if des_prev is not None and des_next is not None:
            matches = bf.match(des_prev, des_next)
            matches = sorted(matches, key=lambda x: x.distance)

            prev_pts = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            next_pts = np.float32([kp_next[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            next_pts_opt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, maxLevel=4)

            if next_pts_opt is None or status is None:
                print("Optical flow failed, using last stabilized frame.")
                stabilized_frame = last_stabilized_frame
                # Add to queue and save to folder
                frame_queue.put(stabilized_frame)
                save_frame_to_folder(stabilized_frame, output_folder, frame_number)
                frame_number += 1
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

                    # Motion threshold to filter small jitters
                    if np.abs(dx) < motion_threshold:
                        dx = 0
                    if np.abs(dy) < motion_threshold:
                        dy = 0
                    if np.abs(da) < motion_threshold:
                        da = 0

                    transform = np.array([dx, dy, da])

                    statics_obj_prev, statics_obj_curr = detect_static_objects(good_prev_pts, good_next_pts, transform)

                    if len(statics_obj_prev) >= 4 and len(statics_obj_curr) >= 4:
                        m2, _ = cv2.estimateAffinePartial2D(statics_obj_prev, statics_obj_curr)

                        # Check if m2 is valid before using it
                        if m2 is not None:
                            dx2 = m2[0, 2]
                            dy2 = m2[1, 2]
                            da2 = np.arctan2(m2[1, 0], m2[0, 0])
                            transform = np.array([dx2, dy2, da2])

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

                    # Apply border fix to handle black borders
                    stabilized_frame = fix_border(stabilized_frame)

                    last_stabilized_frame = stabilized_frame
                else:
                    stabilized_frame = frame
            else:
                print("Not enough valid points for transformation. Using last stabilized frame.")
                stabilized_frame = last_stabilized_frame

            # Add the stabilized frame to the queue and save to folder
            frame_queue.put(stabilized_frame)
            save_frame_to_folder(stabilized_frame, output_folder, frame_number)
            frame_number += 1

            prev_gray = gray
            kp_prev, des_prev = kp_next, des_next

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
    stabilize_video_sift_optical_flow('input2.mp4')
