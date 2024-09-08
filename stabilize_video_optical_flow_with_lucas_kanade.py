import cv2
import numpy as np

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]

def smooth_trajectory(trajectory, radius):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(trajectory.shape[1]):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius)
    return smoothed_trajectory

def fix_border(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def stabilize_video(input_path, output_path, smoothing_radius=50):
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames - 1, 3), np.float32)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.005, minDistance=20, blockSize=3)

    for i in range(n_frames - 2):
        success, curr = cap.read()
        if not success:
            break
        
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        valid_prev_pts = prev_pts[status == 1]
        valid_curr_pts = curr_pts[status == 1]

        if len(valid_prev_pts) < 10:
            print(f"Frame {i}: Not enough points for optical flow. Reinitializing points.")
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.005, minDistance=20, blockSize=3)
            continue

        m, _ = cv2.estimateAffinePartial2D(valid_prev_pts, valid_curr_pts)

        if m is None:
            print(f"Frame {i}: Could not find a valid transformation matrix. Skipping this frame.")
            continue

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        transforms[i] = [dx, dy, da]

        prev_gray = curr_gray
        prev_pts = valid_curr_pts.reshape(-1, 1, 2)

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory, smoothing_radius)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames - 2):
        success, frame = cap.read()
        if not success:
            break

        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        frame_stabilized = fix_border(frame_stabilized)

        out.write(frame_stabilized)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    stabilize_video('input2.mp4', 'stabilized_output_video_optical_flow.mp4')
