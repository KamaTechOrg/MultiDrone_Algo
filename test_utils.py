import find_height_to_test as test
import cv2
import numpy as np
import pytest
image_left = cv2.imread('../im0.png', cv2.IMREAD_GRAYSCALE)
image_right = cv2.imread('../im1.png', cv2.IMREAD_GRAYSCALE)


def generate_test_images(disparity_value, image_size=(100, 100)):
    image_left = np.zeros(image_size, dtype=np.uint8)
    image_right = np.zeros(image_size, dtype=np.uint8)

    # Adding a constant disparity value across the entire image
    if disparity_value > 0:
        image_right[:, int(disparity_value):] = image_left[:, :-int(disparity_value)]
    return image_left, image_right

def test_valid_input():
    x, y = 50, 50
    image_left, image_right = generate_test_images(disparity_value=5)
    focal_length = 1.0
    baseline = 1.0
    camera_height = 10.0

    # Since disparity is constant (e.g., 5), we can calculate expected depth and height.
    disparity_value = 5
    expected_depth = (baseline * focal_length) / disparity_value
    expected_height = camera_height - expected_depth

    height = test.calculate_height_above_ground(x, y, image_left, image_right, focal_length, baseline, camera_height)
    assert height == pytest.approx(expected_height, rel=1e-2)
    
def test_out_of_bounds_coordinates():
    # x, y out of bounds (image size is 100x100)
    result  = test.calculate_height_above_ground(1200,19000, image_left, image_right, focal_length=1733.74, baseline=536.62, camera_height=100)
    assert result is None

def test_missing_image():
    with pytest.raises(ValueError, match="Cannot read one of the images"):
        test.calculate_height_above_ground(120,90, None, None, focal_length=1733.74, baseline=536.62, camera_height=100)

def test_non_integer_coordinates():
    with pytest.raises(TypeError):
        test.calculate_height_above_ground(50.5, 50.5, image_left, image_right, focal_length=1733.74, baseline=536.62, camera_height=100)

def test_negative_focal_length():
    result =test.calculate_height_above_ground(50, 50, image_left, image_right, focal_length=-1733.74, baseline=536.62, camera_height=100)
    print(result)
    assert result is None  # The focal length is invalid, so no height should be calculated
    
def test_camera_at_ground_level():
   result =test.calculate_height_above_ground(50, 50, image_left, image_right, focal_length=1733.74, baseline=536.62, camera_height=0)
   print(result)
   assert  result is None or result < 0# Objects should be below the ground level (negative height)

def test_no_camera_height():
    result =test.calculate_height_above_ground(50, 50, image_left, image_right, focal_length=1733.74, baseline=536.62, camera_height=None)
    print(result)
    assert result is None 

def test_no_baseline():
    result =test.calculate_height_above_ground(50, 50, image_left, image_right, focal_length=1733.74, baseline=None, camera_height=100)
    print(result)
    assert result is None
    
def test_no_focal_length():
    result =test.calculate_height_above_ground(50, 50, image_left, image_right, focal_length=None, baseline=536.62, camera_height=100)
    print(result)
    assert result is None
    
    
# if __name__=="__main__":
#    test_no_focal_length()
    