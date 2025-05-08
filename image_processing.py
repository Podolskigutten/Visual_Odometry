# Test for new file yeeehaaaaa
import cv2
import os
import numpy as np

class ImageLoader:
    def __init__(self, filepath, rate_hz=10):
        self.default_rate = 10  # original rate of images
        self.filepath = filepath
        self.rate_hz = rate_hz
        self.image_files = sorted([
            f for f in os.listdir(filepath)
            if f.lower().endswith(('.png'))
        ])

    def get_indices(self):
        step = int(self.default_rate / self.rate_hz)
        return list(range(0, len(self.image_files), step))

    def load_images(self):
        indices = self.get_indices()
        images = []
        for i in indices:
            full_path = os.path.join(self.filepath, self.image_files[i])
            img = cv2.imread(full_path)
            if img is not None:
                images.append(img)
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
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(images)} images")
                
        return all_features
    


class FeatureMatcher:
    def __init__(self, method='SIFT', ratio_threshold=0.75):
        self.method = method
        self.ratio_threshold = ratio_threshold

    def match_features(self, kp1, des1, kp2, des2, method='SIFT', ratio_threshold=0.75):
        # Create matcher based on the method
        if method == 'SIFT' or method == 'SURF':
            # SIFT and SURF use L2 norm
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            # ORB and AKAZE use Hamming distance
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Match descriptors using kNN
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        
        # Extract coordinates of matched keypoints
        coords1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        coords2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        return coords1, coords2