import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_disparity(img_left, img_right, width, height, ndisp, vmin, vmax, isint):
    # Create stereo block matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=ndisp,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    # Compute disparity map
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    
    # Normalize the disparity map for visualization
    if isint:
        disparity = (disparity - vmin) / (vmax - vmin) * 255
        disparity = np.uint8(disparity)
    
    return disparity

def backproject_to_3d(disparity, cam0, cam1, baseline):
    # Camera matrix for left camera (cam0)
    f = cam0[0, 0]  # Focal length (assuming fx = fy)
    cx = cam0[0, 2]  # Principal point x-coordinate
    cy = cam0[1, 2]  # Principal point y-coordinate

    # Image dimensions
    h, w = disparity.shape

    # Create a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Back-project to 3D coordinates (X, Y, Z)
    Z = baseline * f / (disparity + 1e-6)  # Avoid division by zero
    X = (x - cx) * Z / f
    Y = (y - cy) * Z / f

    # Stack coordinates
    points_3d = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    
    return points_3d

def main(img_left_path, img_right_path, cam0, cam1, baseline, ndisp, vmin, vmax, isint):
    # Load images
    img_left = cv2.imread(img_left_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(img_right_path, cv2.IMREAD_GRAYSCALE)
    
    if img_left is None or img_right is None:
        raise ValueError("Error loading images. Please check the file paths.")
    
    # Compute disparity map
    disparity = compute_disparity(img_left, img_right, width, height, ndisp, vmin, vmax, isint)
    
    # Perform 3D back-projection
    points_3d = backproject_to_3d(disparity, cam0, cam1, baseline)
    
    # Filter out points with invalid depths (large Z values)
    valid_indices = points_3d[:, 2] < 10000
    points_3d_valid = points_3d[valid_indices]
    
    # Save 3D points to a file
    np.savetxt('3D_reconstruction.txt', points_3d_valid, header='X Y Z', comments='')
    
    # Visualize as a scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(points_3d_valid[:, 0], points_3d_valid[:, 1], c=points_3d_valid[:, 2], cmap='hot', s=1, marker='.')
    plt.colorbar(label='Depth (Z-axis)')
    plt.gca().invert_yaxis()
    plt.title("3D Point Cloud Projection")
    plt.xlabel("X (Image Plane)")
    plt.ylabel("Y (Image Plane)")
    plt.savefig('3D_reconstruction_image.png')
    plt.show()

    print("3D reconstruction points saved as 3D_reconstruction.txt and visualization saved as 3D_reconstruction_image.png")

# Camera parameters
cam0 = np.array([[1758.23, 0, 872.36],
                 [0, 1758.23, 552.32],
                 [0, 0, 1]])

cam1 = np.array([[1758.23, 0, 872.36],
                 [0, 1758.23, 552.32],
                 [0, 0, 1]])
# Parameters
baseline =124.86
ndisp = 310
vmin=90
vmax=280
isint = 0
width=1920
height=1080

# Run the main function
main('Middlebury/data/chess2/im0.png', 'Middlebury/data/chess2/im1.png', cam0, cam1, baseline, ndisp, vmin, vmax, isint)
