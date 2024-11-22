import cv2
import numpy as np

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size)/window_size
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

    # Set up the output video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Initialize Farneback optical flow
    farneback_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Compute optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **farneback_params)
        
        # Compute the transformation matrix
        dx, dy = np.mean(flow[..., 0]), np.mean(flow[..., 1])
        da = 0  # Rotation is not estimated with Farneback; you may need additional methods if rotation is important
        
        transforms[i] = [dx, dy, da]

        prev_gray = curr_gray

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

        # Reconstruct transformation matrix
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = 1
        m[1, 1] = 1
        m[0, 2] = dx
        m[1, 2] = dy

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        frame_stabilized = fix_border(frame_stabilized)

        # Write stabilized frame to output video
        out.write(frame_stabilized)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    stabilize_video('input2.mp4', 'stabilized_output_video_farneback.mp4')
