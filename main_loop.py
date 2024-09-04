import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import normalized_mutual_info_score
from PIL import Image
from pathlib import Path


class ImageProcessor:
    def __init__(self, method='brisk'):
        self.method = method

    def detect_and_compute(self, img):
        match self.method:
            case 'brisk':
                return self._detect_and_compute_brisk(img)
            case 'sift':
                return self._detect_and_compute_sift(img)
            case 'orb':
                return self._detect_and_compute_orb(img)
            case 'akaze':
                return self._detect_and_compute_akaze(img)
            case _:
                raise ValueError("Unknown method")

    def _detect_and_compute_brisk(self, img):
        brisk = cv2.BRISK_create()
        keypoints, descriptors = brisk.detectAndCompute(img, None)
        return keypoints, descriptors

    def _detect_and_compute_sift(self, img):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        return keypoints, descriptors

    def _detect_and_compute_orb(self, img):
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(img, None)
        return keypoints, descriptors

    def _detect_and_compute_akaze(self, img):
        akaze = cv2.AKAZE_create()
        keypoints, descriptors = akaze.detectAndCompute(img, None)
        return keypoints, descriptors

    def match_features(self, des1, des2):
        if self.method in ['brisk', 'orb', 'akaze']:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
        elif self.method == 'sift':
            flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            matches = flann.knnMatch(des1, des2, k=2)
            matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        else:
            raise ValueError("Unknown method")
        return matches

    def find_homography(self, kp1, kp2, matches):
        if len(matches) < 4:
            raise ValueError("Not enough matches found to compute homography")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        inliers = np.sum(mask) if H is not None else 0
        return H, inliers

    def warp_image(self, img, H, shape):
        return cv2.warpPerspective(img, H, shape)

    def calculate_ssim(self, img1, img2):
        return ssim(img1, img2)

    def calculate_nmi(self, img1, img2):
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        return normalized_mutual_info_score(img1_flat, img2_flat)

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Switch to PIL")
            return img
        except ValueError:
            try:
                img = Image.open(image_path).convert('L')
                img = np.array(img)
                return img
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}")

    def visualize_matches(self, img1, img2, kp1, kp2, matches):
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def register_images(self, img1_path, img2_path):
        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)

        kp1, des1 = self.detect_and_compute(img1)
        kp2, des2 = self.detect_and_compute(img2)

        matches = self.match_features(des1, des2)

        if len(matches) < 4:
            raise ValueError("Not enough matches found to compute homography")

        H, inliers = self.find_homography(kp1, kp2, matches)

        warped_img = self.warp_image(img2, H, (img1.shape[1], img1.shape[0]))

        ssim_value = self.calculate_ssim(img1, warped_img)
        nmi_value = self.calculate_nmi(img1, warped_img)

        num_keypoints1 = len(kp1)
        num_keypoints2 = len(kp2)
        num_matches = len(matches)
        inliers_ratio = inliers / num_matches if num_matches > 0 else 0

        # visualize_matches(img1, img2, kp1, kp2, matches)

        return inliers_ratio, ssim_value, nmi_value, warped_img

    def register_images_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        np_ssim = []
        np_nmi = []
        np_inliers_ratio = []

        for index, row in df.iterrows():
            img1_path = row['query_name']
            img2_path = row['ref_name']

            self.verify_file_exists(img1_path)
            self.verify_file_exists(img2_path)

            try:
                inliers_ratio, ssim_value, nmi_value, warped_img = self.register_images(img1_path, img2_path)
                np_inliers_ratio.append(inliers_ratio)
                np_ssim.append(ssim_value)
                np_nmi.append(nmi_value)

                output_path = Path('warp') / f'warped_{index}.png'
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), warped_img)
            except Exception as e:
                print(f"Error processing images at row {index}: {e}")

        return np_inliers_ratio, np_ssim, np_nmi

    def calculate_statistics(self, nmi, ssim, np_inliers_ratio):
        nmi_np = np.array(nmi)
        ssim_np = np.array(ssim)
        inliers_ratio_np = np.array(np_inliers_ratio)

        stats = {
            "NMI": {
                "mean": np.mean(nmi_np).round(4),
                "median": np.median(nmi_np).round(4),
                "min": np.min(nmi_np).round(4),
                "max": np.max(nmi_np).round(4)
            },
            "SSIM": {
                "mean": np.mean(ssim_np).round(4),
                "median": np.median(ssim_np).round(4),
                "min": np.min(ssim_np).round(4),
                "max": np.max(ssim_np).round(4)
            },
            "Inliers Ratio": {
                "mean": np.mean(inliers_ratio_np).round(4),
                "median": np.median(inliers_ratio_np).round(4),
                "min": np.min(inliers_ratio_np).round(4),
                "max": np.max(inliers_ratio_np).round(4)
            }
        }

        return stats

    def verify_file_exists(self, file_path):
        pass
        #if os.path.exists(file_path):
            #print(f"File exists: {file_path}")
        #else:
            #print(f"File not found: {file_path}")


if __name__ == '__main__':
    print('orb')
    processor = ImageProcessor(method='orb')
    np_inliers_ratio, np_ssim, np_nmi = processor.register_images_from_csv('image_pairs.csv')
    print(processor.calculate_statistics(np_nmi, np_ssim, np_inliers_ratio))
    print('sift')
    processor = ImageProcessor(method='sift')
    np_inliers_ratio, np_ssim, np_nmi = processor.register_images_from_csv('image_pairs.csv')
    print(processor.calculate_statistics(np_nmi, np_ssim, np_inliers_ratio))
    print('brisk')
    processor = ImageProcessor(method='brisk')
    np_inliers_ratio, np_ssim, np_nmi = processor.register_images_from_csv('image_pairs.csv')
    print(processor.calculate_statistics(np_nmi, np_ssim, np_inliers_ratio))
    print('akaze')
    processor = ImageProcessor(method='akaze')
    np_inliers_ratio, np_ssim, np_nmi = processor.register_images_from_csv('image_pairs.csv')
    print(processor.calculate_statistics(np_nmi, np_ssim, np_inliers_ratio))
