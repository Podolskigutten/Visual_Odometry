import os
import numpy as np
import cv2
from image_processing_2 import ImageLoader, FeatureDetecor, FeatureMatcher
from motion_plot_2 import read_ground_truth_positions, estimate_motion_from_correspondences, plot_with_estimated_motion


def main():
    # File paths
    print("OpenCV version:", cv2.__version__)
    ground_truth_file = 'Images/poses_ground_truth/00.txt'
    image_folder = 'Images/00/image_0/'
    ground_truth_positions = read_ground_truth_positions(ground_truth_file)

    # Camera intrinsics (KITTI)
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])

    # Load images
    path = os.path.join("Images", "00", "image_0")
    loader = ImageLoader(path)  # Load all available images
    images = loader.load_images()
    print(f"Loaded {len(images)} images")

    # Choose feature detection method
    method = 'SIFT'  # Choose between SIFT, ORB and AKAZE

    # Detect features in all images
    detector = FeatureDetecor(method)
    features = detector.detect_all_features(images)
    print(f"Extracted features from {len(features)} images")

    # Feature matching setup
    matcher = FeatureMatcher(method)

    # Initialize motion tracking variables
    R_total = np.eye(3)  # Identity rotation matrix
    t_total = np.zeros((3, 1))  # Zero translation vector

    # Scale factor for visualization (may need adjustment)
    initial_scale = 0.75  # Increased scale factor to make motion more visible

    # Create visualization window
    cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trajectory', 800, 800)

    # Process consecutive image pairs
    for i in range(len(features) - 1):
        # Get current pair of features
        kp1, des1 = features[i]
        kp2, des2 = features[i + 1]

        # Match features between consecutive images
        coords1, coords2 = matcher.match_features(kp1, des1, kp2, des2)
        num_matches = len(coords1)
        print(f"Good matches between frames {i} and {i + 1}: {num_matches}")

        # Skip if too few matches
        if num_matches < 8:
            print(f"Warning: Too few matches ({num_matches}) to estimate motion reliably. Skipping frame {i + 1}.")
            continue

        # Estimate motion from matched feature coordinates
        R, t = estimate_motion_from_correspondences(coords1, coords2, K)

        # IMPORTANT: Update rotation first, then translation
        # For visual odometry, we need to invert the motion (since the camera moves opposite to perceived motion)
        R_inv = R.T
        t_inv = -R_inv @ t

        # Update accumulated motion
        R_total = R_inv @ R_total  # Update total rotation
        t_total = t_total + initial_scale * (R_total @ t_inv)  # Apply rotation to translation before accumulating

        # Print current motion estimate
        print(f"Frame {i + 1}: Translation (x,y,z): ({t_total[0, 0]:.3f}, {t_total[1, 0]:.3f}, {t_total[2, 0]:.3f})")

        # Visualize current trajectory
        key = plot_with_estimated_motion(
            ground_truth_positions[:i + 2],  # Only plot up to current frame
            R_total,
            t_total,
            image_folder,
            K,
            max_frames=i + 2
        )

        # Check for ESC key to exit
        if key == 27:  # ESC key
            print("ESC pressed. Exiting...")
            break
    # Check for ESC key to exit
    print("\nProcessing complete. Press ESC to exit.")
    while True:
        key = cv2.waitKey(0)  # Wait indefinitely for a key press
        if key == 27:  # ESC key
            print("ESC pressed. Exiting...")
            break
    cv2.destroyAllWindows()

   


if __name__ == "__main__":
    main()