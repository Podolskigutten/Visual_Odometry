# Test for new file 2
import cv2
import os
import numpy as np


class ImageLoader:
    def __init__(self, filepath, max_images=None):
        self.default_rate = 10  # original rate of images
        self.filepath = filepath
        self.max_images = max_images
        self.image_files = sorted([
            f for f in os.listdir(filepath)
            if f.lower().endswith(('.png'))
        ])

        # Limit the number of images if specified
        if max_images is not None:
            self.image_files = self.image_files[:max_images]

    def load_images(self):
        images = []
        for i, img_file in enumerate(self.image_files):
            full_path = os.path.join(self.filepath, img_file)
            img = cv2.imread(full_path)
            if img is not None:
                images.append(img)

            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/{len(self.image_files)} images")

        return images


class FeatureDetecor:
    def __init__(self, method='SIFT'):
        self.method = method
        if self.method == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=1500)
        elif method == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=1500)
        elif method == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported method: {method}")

    def detect_all_features(self, images):
        """Detect features in all images.
        Args:
            images: List of images
        Returns:
            List of (keypoints, descriptors) tuples
        """
        all_features = []
        for i, img in enumerate(images):
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            all_features.append((keypoints, descriptors))

            # Optional: Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(images)} images")

        return all_features


class FeatureMatcher:
    def __init__(self, method='SIFT', ratio_threshold=0.75):
        self.method = method
        self.ratio_threshold = ratio_threshold

    def match_features(self, kp1, des1, kp2, des2):
        # Create matcher based on the method
        if self.method == 'SIFT' or self.method == 'SURF':
            # SIFT and SURF use L2 norm
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            # ORB and AKAZE use Hamming distance
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Match descriptors using kNN
        matches = matcher.knnMatch(des1, des2, k=2)

        # Apply ratio test to find good matches
        good_matches = []
        for pair in matches:
            if len(pair) < 2:  # Handle cases where fewer than k matches are found
                continue

            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        # Extract coordinates of matched keypoints
        if len(good_matches) >= 8:  # Minimum needed for Essential matrix
            coords1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            coords2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            return coords1, coords2
        else:
            return np.array([]), np.array([])