import numpy as np
import cv2
import open3d as o3d


def load_images(left_image_path, right_image_path, color_image_path):
    """Load the stereo and color images from the specified paths."""
    img_left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    colors = cv2.imread(color_image_path)

    if img_left is None or img_right is None or colors is None:
        raise IOError("Could not load one or more images. Please check the file paths.")

    return img_left, img_right, colors


def compute_disparity(img_left, img_right, vmin, ndisp):
    """Compute the disparity map using StereoSGBM algorithm."""
    stereo = cv2.StereoSGBM_create(
        minDisparity=vmin,
        numDisparities=ndisp,
        blockSize=15,
        P1=8 * 3 * 3 ** 2,
        P2=32 * 3 * 3 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=2
    )
    disparity_map = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    return disparity_map


def filter_disparity(disparity_map, vmin, vmax):
    """Filter out points with low or high disparity values (background)."""
    return (disparity_map > vmin) & (disparity_map < vmax)


def create_q_matrix(cam0, baseline, doffs):
    """Create the Q matrix based on camera parameters."""
    return np.float32([
        [1, 0, 0, -cam0[0, 2]],
        [0, -1, 0, cam0[1, 2]],
        [0, 0, 0, cam0[0, 0]],
        [0, 0, 1 / baseline, doffs]
    ])


def create_point_cloud(disparity_map, Q, mask, colors):
    """Convert the disparity map to a 3D point cloud and apply color filtering."""
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    output_points = points_3d[mask]

    colors_rgb = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    colors_filtered = colors_rgb[mask]

    if output_points.shape[0] != colors_filtered.shape[0]:
        raise ValueError("Mismatch between points and color data.")

    return output_points, colors_filtered


def visualize_point_cloud(output_points, colors_filtered):
    """Visualize the 3D point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points)
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered[:, :3] / 255.0)  # Normalize colors
    o3d.visualization.draw_geometries([pcd])


def main():
    left_image_path = r'C:\BootCamp\project\01.09\3D_reconstruction\cloud\artroom\im0.png'
    right_image_path = r'C:\BootCamp\project\01.09\3D_reconstruction\cloud\artroom\im1.png'
    color_image_path = r'C:\BootCamp\project\01.09\3D_reconstruction\cloud\artroom\im0.png'

    img_left, img_right, colors = load_images(left_image_path, right_image_path, color_image_path)

    # Camera intrinsic parameters (calibration matrices)
    cam0 = np.array([[1733.74, 0, 792.27], [0, 1733.74, 541.89], [0, 0, 1]])

    # Stereo parameters
    baseline = 536.62
    doffs = 1
    vmin = 55
    vmax = 142
    ndisp = 170

    disparity_map = compute_disparity(img_left, img_right, vmin, ndisp)

    mask = filter_disparity(disparity_map, vmin, vmax)

    Q = create_q_matrix(cam0, baseline, doffs)

    # Create 3D point cloud
    output_points, colors_filtered = create_point_cloud(disparity_map, Q, mask, colors)

    # Visualize the 3D point cloud
    visualize_point_cloud(output_points, colors_filtered)


if __name__ == "__main__":
    main()
