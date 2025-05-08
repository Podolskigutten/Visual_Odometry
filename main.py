import os
import numpy as np
import matplotlib as plt
import cv2 
from image_processing import ImageLoader, FeatureDetecor, FeatureMatcher
from Motion_plot import read_ground_truth_positions, estimate_motion_from_correspondences, plot_with_estimated_motion



def main():
    ground_truth_file = 'Images/poses_ground_truth/00.txt'
    image_folder = 'Images/00/image_0/'
    ground_truth_positions = read_ground_truth_positions(ground_truth_file)

    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])

    #start by loading images
    path = os.path.join("Images", "00", "image_0")

    loader = ImageLoader(path, 2)
    images = loader.load_images()
    print(f"Loaded {len(images)} images")

    # Choose feature detection method
    method = 'SIFT'  # Choose between SIFT, ORB and AKAZE

    # Detect features in all images, results in keypoints and descriptors
    detector = FeatureDetecor(method) # Choose between SIFT, ORB and AKAZE
    features = detector.detect_all_features(images)

    # Pairwise match features, esitmate essential matrix 
    # and calculate the translation between image pairs
    matcher = FeatureMatcher(method)

    # Initialize the rotation (R) and translation (t) as identity (for visualization)
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    initial_scale = 1.0

    # Process consecutive image pairs
    for i in range(len(features) - 1):
        # Get current pair of features
        kp1, des1 = features[i]
        kp2, des2 = features[i + 1]
        
        # Match features between consecutive images
        coords1, coords2 = matcher.match_features(kp1, des1, kp2, des2)
        print(f"Good matches between frames {i} and {i + 1}: {len(coords1)}")

        # Now comes your code broooooo, estimate the shit out of the Essential matrix
        R, t = estimate_motion_from_correspondences(coords1, coords2, K)

        # Update the total motion (accumulate)
        t_total += initial_scale * (R_total @ t)

        R_total = R_total @ R

        key = plot_with_estimated_motion(ground_truth_positions, R_total, t_total, image_folder, K, max_frames=1000)

        if key == 27:  # ESC key
            print("ESC pressed. Exiting...")
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()