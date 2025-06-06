# Test for new file 2
import cv2
import os
import numpy as np
import h5py

def get_path_and_intrinsic(dataset_num):
    # Valid dataset numbers
    valid_datasets = [0, 1, 3, 5, 9]
    
    if dataset_num not in valid_datasets:
        raise ValueError(f"Invalid dataset number. Choose from: {valid_datasets}")
    
    # Format dataset number as two digits
    dataset_str = f"{dataset_num:02d}"
    
    print(f"Data set {dataset_str} chosen")
    
    # Define paths
    path_images = os.path.join("Images", dataset_str, "image_0")
    path_ground_truth = f'Images/poses_ground_truth/{dataset_str}.txt'
    path_calib = os.path.join("Images", dataset_str, "calib.txt")
    
    # Read calibration file and extract P0 matrix
    try:
        with open(path_calib, 'r') as f:
            lines = f.readlines()
            
        # Find the P0 line
        p0_line = None
        for line in lines:
            if line.startswith("P0:"):
                p0_line = line
                break
        
        if p0_line is None:
            raise ValueError(f"P0 matrix not found in {path_calib}")
        
        # Extract values after "P0:"
        p0_values = p0_line.split()[1:]  # Skip "P0:" and get the rest
        p0_values = [float(val) for val in p0_values]
        
        # Convert to 3x4 matrix
        P0 = np.array(p0_values).reshape(3, 4)
        
        # Extract 3x3 intrinsic matrix (remove last column)
        K = P0[:, :3]
        
    except FileNotFoundError:
        print(f"Warning: Calibration file not found at {path_calib}")
        print("Using default KITTI intrinsics")
        # Default KITTI intrinsics as fallback
        K = np.array([[718.856, 0, 607.1928],
                     [0, 718.856, 185.2157],
                     [0, 0, 1]])
    
    return path_images, path_ground_truth, K

def load_features_from_hdf5(filename):
    """Load features from HDF5 file in the same format as FeatureDetector.detect_all_features()"""
    
    features = []
    
    with h5py.File(filename, 'r') as f:
        num_images = f.attrs['num_images']
        
        for i in range(num_images):
            img_key = f'image_{i:06d}'
            if img_key in f:
                kp_data = f[img_key]['keypoints'][:]
                desc_data = f[img_key]['descriptors'][:]
                
                if len(kp_data) > 0:
                    # Convert back to cv2.KeyPoint objects
                    keypoints = []
                    for kp_row in kp_data:
                        kp = cv2.KeyPoint(x=kp_row[0], y=kp_row[1], size=1, 
                                        angle=kp_row[2], response=kp_row[3], 
                                        octave=int(kp_row[4]), class_id=int(kp_row[5]))
                        keypoints.append(kp)
                    
                    descriptors = desc_data if len(desc_data) > 0 else None
                else:
                    keypoints = []
                    descriptors = None
                    
                features.append((keypoints, descriptors))
            else:
                # Handle missing image data
                features.append(([], None))
    
    return features

class ImageLoader:
    def __init__(self, path_image, path_ground_truth, desired_rate=None, max_images=None):
        self.default_rate = 10  # original frequency (Hz)
        self.image_path = path_image
        self.ground_truth_path = path_ground_truth
        self.max_images = max_images

        # Load all .png files
        all_files = sorted([
            f for f in os.listdir(path_image)
            if f.lower().endswith('.png')
        ])

        # Load all ground truth positions (before subsampling)
        self.all_ground_truth = self._load_all_ground_truth()

        # Subsample based on desired rate
        if desired_rate is not None and desired_rate < self.default_rate:
            step = int(self.default_rate / desired_rate)
            self.image_files = all_files[::step]
            # Also subsample ground truth indices
            self.selected_indices = list(range(0, len(all_files), step))
        else:
            self.image_files = all_files
            self.selected_indices = list(range(len(all_files)))

        # Limit number of images if specified
        if max_images is not None:
            self.image_files = self.image_files[:max_images]
            self.selected_indices = self.selected_indices[:max_images]

    def _load_all_ground_truth(self):
        """Load all ground truth positions from file"""
        positions = []
        with open(self.ground_truth_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) == 12:
                    try:
                        x = float(values[3])  # 4th value (index 3)
                        z = float(values[11])  # 12th value (index 11)
                        positions.append((x, z))
                    except ValueError:
                        # If there's an error, append None to maintain sync
                        positions.append(None)
                else:
                    # If row doesn't have 12 values, append None
                    positions.append(None)
        return positions

    def load_images(self):
        images = []
        ground_truth_positions = []

        for i, (img_file, gt_idx) in enumerate(zip(self.image_files, self.selected_indices)):
            # Load image
            full_path = os.path.join(self.image_path, img_file)
            img = cv2.imread(full_path)

            if img is not None:
                images.append(img)

                # Get corresponding ground truth (check bounds and None values)
                if gt_idx < len(self.all_ground_truth) and self.all_ground_truth[gt_idx] is not None:
                    ground_truth_positions.append(self.all_ground_truth[gt_idx])
                else:
                    # Skip this image if ground truth is invalid
                    images.pop()  # Remove the image we just added
                    print(f"Warning: Skipping image {img_file} due to missing ground truth")
                    continue

            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/{len(self.image_files)} images")

        print(f"Successfully loaded {len(images)} images with synchronized ground truth")
        return images, ground_truth_positions


class FeatureDetector:
    def __init__(self, method='SIFT'):
        self.method = method
        if self.method == 'SIFT':
            # More features, better contrast threshold
            self.detector = cv2.SIFT_create(
                nfeatures=2000,  # Increased from 1500
                contrastThreshold=0.03,  # Better for low-contrast features
                edgeThreshold=10,  # Standard value
                sigma=1.6  # Standard value
            )
        elif method == 'ORB':
            self.detector = cv2.ORB_create(
                nfeatures=2000,  # Increased
                scaleFactor=1.2,  # Pyramid scale
                nlevels=8,  # More scale levels
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=20
            )
        elif method == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported method: {method}")

    def detect_all_features(self, images):
        all_features = []
        for i, img in enumerate(images):
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # Apply CLAHE for better contrast (helps with features)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Optional: Apply slight Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (3, 3), 0.5)

            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            # Sort keypoints by response (strength) and keep best ones
            if len(keypoints) > 2000:  # Limit max keypoints
                keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:2000]
                # Recompute descriptors for selected keypoints
                keypoints, descriptors = self.detector.compute(gray, keypoints)

            all_features.append((keypoints, descriptors))

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(images)} images with {len(keypoints)} features")

        return all_features


class FeatureMatcher:
    def __init__(self, method='SIFT', ratio_threshold=0.7):  # Lowered from 0.75
        self.method = method
        self.ratio_threshold = ratio_threshold

    def match_features(self, kp1, des1, kp2, des2):
        if self.method in ['SIFT', 'SURF']:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Add FLANN-based matcher option for better performance with SIFT
        if self.method == 'SIFT' and des1.shape[0] > 500:  # Use FLANN for large sets
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)

        matches = matcher.knnMatch(des1, des2, k=2)

        # More robust ratio test
        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        # Add symmetry check (optional but helps)
        if len(good_matches) >= 8:
            coords1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            coords2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            # Additional geometric verification using fundamental matrix
            F, mask = cv2.findFundamentalMat(coords1, coords2, cv2.FM_RANSAC, 3.0)
            if mask is not None:
                mask = mask.ravel().astype(bool)
                coords1 = coords1[mask]
                coords2 = coords2[mask]

            return coords1, coords2
        else:
            return np.array([]), np.array([])