import cv2
import numpy as np
import queue
import os

# Initialize a global queue for stabilized frames
frame_queue = queue.Queue()


# Function to save each frame to a specified folder
def save_frame_to_folder(frame, folder_path, frame_number):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it doesn't exist
    frame_filename = os.path.join(folder_path, f"frame_{frame_number:04d}.png")
    cv2.imwrite(frame_filename, frame)  # Save the frame as an image


# Weighted moving average smoothing function
def weighted_moving_average(curve, radius):
    if radius < 0:
        return curve
    if radius == 0 or len(curve) == 0:
        return curve

    weights = np.linspace(1, 0, 2 * radius + 1)
    weights /= np.sum(weights)

    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')

    curve_smoothed = np.convolve(curve_pad, weights, mode='same')

    if radius >= len(curve):
        return np.full_like(curve, np.mean(curve, dtype=np.float64), dtype=np.float64)

    return curve_smoothed[radius:-radius]


# Function to fix the borders of the stabilized frame
def fix_border(frame):
    if frame is None or frame.size == 0:
        print("Error: Invalid frame passed to border fix.")
        return frame  # Edge case check: handle empty or invalid frames

    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.05)
    return cv2.warpAffine(frame, T, (s[1], s[0]))


# Function to detect static objects by comparing frame-to-frame movements
def detect_static_objects(prev_pts, curr_pts, camera_transform, threshold=2.0):
    dx_camera, dy_camera, da_camera = camera_transform
    object_displacements = np.linalg.norm(curr_pts - prev_pts, axis=1)  # Compute object movements
    camera_displacement = np.linalg.norm([dx_camera, dy_camera])

    static_points_prev = []
    static_points_curr = []

    for i, displacement in enumerate(object_displacements):
        # Static points have lower displacement than the camera
        if abs(displacement) <= abs(camera_displacement) or displacement < threshold:
            static_points_prev.append(prev_pts[i])
            static_points_curr.append(curr_pts[i])

    # Return arrays of static points (for transformation estimation)
    return np.array(static_points_prev), np.array(static_points_curr)


# Main stabilization function using SIFT and optical flow
def stabilize_video_sift_optical_flow(input_path, output_folder='stabilized_frames', smoothing_radius=50,
                                      motion_threshold=0.15):
    if not os.path.isfile(input_path):
        print(f"Error: Input video file '{input_path}' does not exist.")  # Edge case: invalid input file
        return

    cap = cv2.VideoCapture(input_path)
    frame_number = 0

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize SIFT detector and BFMatcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame of the video.")
        cap.release()
        return  # Exit if the video couldn't be read

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = sift.detectAndCompute(prev_gray, None)

    transforms = []
    trajectory = []
    smoothed_trajectory = np.zeros((3,), np.float32)
    last_stabilized_frame = prev_frame.copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop when there are no more frames to read

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_next, des_next = sift.detectAndCompute(gray, None)

        # Edge case: Check if descriptors were found
        if des_prev is not None and des_next is not None:
            matches = bf.match(des_prev, des_next)
            matches = sorted(matches, key=lambda x: x.distance)

            # Ensure sufficient matches are found
            if len(matches) < 4:
                print(f"Error: Not enough feature matches. Skipping frame {frame_number}.")
                stabilized_frame = last_stabilized_frame
            else:
                prev_pts = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                next_pts = np.float32([kp_next[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Calculate optical flow for improved tracking
                next_pts_opt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, maxLevel=4)

                # Edge case: Optical flow failure
                if next_pts_opt is None or status is None:
                    print(f"Optical flow failed at frame {frame_number}.")
                    stabilized_frame = last_stabilized_frame
                else:
                    good_prev_pts = prev_pts[status == 1]
                    good_next_pts = next_pts_opt[status == 1]

                    if len(good_prev_pts) >= 4 and len(good_next_pts) >= 4:
                        H, inliers = cv2.estimateAffinePartial2D(good_prev_pts, good_next_pts, method=cv2.RANSAC,
                                                                 ransacReprojThreshold=3)

                        # Check if homography is valid
                        if H is None:
                            print(f"Homography estimation failed at frame {frame_number}. Using last stabilized frame.")
                            stabilized_frame = last_stabilized_frame
                        else:
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

                            # Detect static objects and refine transformation
                            transform = np.array([dx, dy, da])
                            statics_obj_prev, statics_obj_curr = detect_static_objects(good_prev_pts, good_next_pts,
                                                                                       transform)

                            if len(statics_obj_prev) >= 4 and len(statics_obj_curr) >= 4:
                                m2, _ = cv2.estimateAffinePartial2D(statics_obj_prev, statics_obj_curr)
                                if m2 is not None:
                                    dx2 = m2[0, 2]
                                    dy2 = m2[1, 2]
                                    da2 = np.arctan2(m2[1, 0], m2[0, 0])
                                    transform = np.array([dx2, dy2, da2])

                            # Add to transformations and trajectory lists
                            transforms.append(transform)
                            trajectory.append(transform)

                            # Smooth trajectory if enough frames
                            if len(transforms) > smoothing_radius:
                                smoothed_trajectory = np.mean(transforms[-smoothing_radius:], axis=0)

                            difference = smoothed_trajectory - np.sum(trajectory, axis=0)
                            dx += difference[0]
                            dy += difference[1]
                            da += difference[2]

                            # Apply the corrected transformation
                            H_corrected = np.array([
                                [np.cos(da), -np.sin(da), dx],
                                [np.sin(da), np.cos(da), dy]
                            ], dtype=np.float32)

                            stabilized_frame = cv2.warpAffine(frame, H_corrected, (frame.shape[1], frame.shape[0]))

                            # Fix borders after stabilization
                            stabilized_frame = fix_border(stabilized_frame)
                            last_stabilized_frame = stabilized_frame
                    else:
                        print(f"Insufficient valid points for transformation at frame {frame_number}.")
                        stabilized_frame = last_stabilized_frame
        else:
            print(f"Descriptors missing at frame {frame_number}. Using last stabilized frame.")
            stabilized_frame = last_stabilized_frame

        # Save the stabilized frame to the queue and output folder
        frame_queue.put(stabilized_frame)
        save_frame_to_folder(stabilized_frame, output_folder, frame_number)
        frame_number += 1

        # Update previous frame data
        prev_gray = gray
        kp_prev, des_prev = kp_next, des_next

    cap.release()
    cv2.destroyAllWindows()


# Function to retrieve stabilized frames from the queue
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
