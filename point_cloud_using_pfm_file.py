import numpy as np
import cv2
import open3d as o3d
import re


def load_pfm(file_path):
    """Load a PFM file and return the image and scale."""
    with open(file_path, 'rb') as file:
        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if not dim_match:
            raise ValueError('Malformed PFM header.')

        width, height = map(int, dim_match.groups())
        scale = float(file.readline().decode('utf-8').rstrip())
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flip(data, 1)

        return data, scale


def load_images(left_image_path, right_image_path, color_image_path):
    """Load the stereo and color images from the specified paths."""
    img_left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    colors = cv2.imread(color_image_path)

    if img_left is None or img_right is None or colors is None:
        raise IOError("Could not load one or more images. Please check the file paths.")

    img_left = cv2.flip(img_left, -1)
    img_right = cv2.flip(img_right, -1)
    colors = cv2.flip(colors, -1)

    return img_left, img_right, colors


def filter_disparity(disparity_map, vmin, vmax):
    """Filter out points with low or high disparity
    values (background)."""
    return (disparity_map > vmin) & (disparity_map < vmax)


def create_q_matrix(cam0, baseline, doffs):
    """Create the Q matrix based on camera
    parameters."""
    return np.float32([
        [1, 0, 0, -cam0[0, 2]],
        [0, -1, 0, cam0[1, 2]],
        [0, 0, 0, cam0[0, 0]],
        [0, 0, 1 / baseline, doffs]
    ])


def create_point_cloud(disparity_map, Q, mask, colors):
    """Convert the disparity map to a 3D point cloud and apply
    color filtering."""
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
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered[:, :3] / 255.0)
    o3d.visualization.draw_geometries([pcd])


def main():
    directory_path = r'C:\BootCamp\project\01.09\3D_reconstruction\cloud'
    left_image_path = f'{directory_path}\\artroom\\im0.png'
    right_image_path = f'{directory_path}\\artroom\\im1.png'
    color_image_path = f'{directory_path}\\artroom\\im0.png'
    pfm_file_path = f'{directory_path}\\artroom\\disp0.pfm'

    # Load the PFM disparity map
    disparity_map, scale = load_pfm(pfm_file_path)
    img_left, img_right, colors = load_images(left_image_path, right_image_path, color_image_path)

    cam0 = np.array([[1733.74, 0, 792.27], [0, 1733.74, 541.89], [0, 0, 1]])

    # Stereo parameters
    baseline = -536.62
    doffs = 0
    vmin = 55
    vmax = 142

    # Filter the disparity map
    mask = filter_disparity(disparity_map, vmin, vmax)

    # Create the Q matrix
    Q = create_q_matrix(cam0, baseline, doffs)

    # Create 3D point cloud
    output_points, colors_filtered = create_point_cloud(disparity_map, Q, mask, colors)
    visualize_point_cloud(output_points, colors_filtered)

if __name__ == "__main__":
    main()
