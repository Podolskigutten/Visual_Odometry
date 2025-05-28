import os
import numpy as np
import cv2
from image_processing import ImageLoader, FeatureDetector, FeatureMatcher, get_path_and_intrinsic, load_features_from_hdf5
from motion_plot import estimate_motion_from_correspondences, plot_with_estimated_motion

def main():
    dataset_num = 9  # Change this to 0, 1, 2, 3, 5, or 9 as needed

    # Load the chosen dataset
    path_images, path_ground_truth, K = get_path_and_intrinsic(dataset_num)

    # Continue with the rest of your code using these variables
    print(f"Using dataset {dataset_num:02d}")
    print(f"Images path: {path_images}")
    print(f"Ground truth path: {path_ground_truth}")
    print(f"Camera intrinsics:\n{K}")

    rate = 10
    loader = ImageLoader(path_images, path_ground_truth, desired_rate=rate)
    images, ground_truth_positions = loader.load_images()
    print(f"Loaded {len(images)} images")

    # Choose feature detection method
    method = 'SIFT'

    # Try to load pre-computed features
    features_filename = f"Sequence {dataset_num:02d} features {method}.h5"
    
    if os.path.exists(features_filename):
        print(f"Loading pre-computed features from {features_filename}")
        features = load_features_from_hdf5(features_filename)
        print(f"Loaded features from {len(features)} images")
    else:
        print(f"Pre-computed features not found ({features_filename})")
        print("Computing features on-the-fly...")
        # Fallback to original method
        detector = FeatureDetector(method)
        features = detector.detect_all_features(images)
        print(f"Extracted features from {len(features)} images")

    # Feature matching setup
    matcher = FeatureMatcher(method)

    # Initialize motion tracking variables
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    # Keyframe selection parameters
    min_inliers = 50  # Minimum inliers for reliable motion estimation
    min_translation = 0.65  # Minimum translation norm to consider a keyframe
    last_keyframe_idx = 0
    keyframes = [0]  # Store keyframe indices


    # Scale factor for visualization
    if dataset_num == 0:
        initial_scale = 0.75 * (10/rate)
    elif dataset_num == 1:
        initial_scale = 2.4 * (10/rate)
    elif dataset_num == 3:
        initial_scale = .7 * (10/rate)
    elif dataset_num == 5:
        initial_scale = .8 * (10/rate)
    elif dataset_num == 9:
        initial_scale = 1 * (10/rate)
    else:
        initial_scale = 1 * (10/rate)


    # Create visualization window
    cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trajectory', 800, 800)


    # Process image pairs
    for i in range(len(features) - 1):
        # Get current pair of features (compare with last keyframe)
        kp1, des1 = features[last_keyframe_idx]
        kp2, des2 = features[i + 1]

        # Match features between last keyframe and current frame
        coords1, coords2 = matcher.match_features(kp1, des1, kp2, des2)
        num_matches = len(coords1)
        print(f"Good matches between frames {last_keyframe_idx} and {i + 1}: {num_matches}")

        # Skip if too few matches
        if num_matches < 8:
            print(f"Warning: Too few matches ({num_matches}). Skipping frame {i + 1}.")
            continue

        # Estimate motion
        R, t, inlier_pts1, inlier_pts2 = estimate_motion_from_correspondences(coords1, coords2, K)

        # Check for sufficient inliers
        if len(inlier_pts1) < min_inliers:
            print(f"Warning: Too few inliers ({len(inlier_pts1)} < {min_inliers}). Skipping frame {i + 1}.")
            continue

        # Check for sufficient translation (to detect stationary car)
        translation_norm = np.linalg.norm(t)
        if translation_norm < min_translation:
            print(f"Warning: Negligible motion (translation norm: {translation_norm:.3f} < {min_translation}). Skipping frame {i + 1}.")
            continue

        # Frame qualifies as a keyframe
        print(f"Frame {i + 1} selected as keyframe (translation norm: {translation_norm:.3f})")
        keyframes.append(i + 1)
        last_keyframe_idx = i + 1

        # Update motion (invert as before)
        R_inv = R.T
        t_inv = -R_inv @ t
        R_total = R_inv @ R_total
        t_total = t_total + initial_scale * (R_total @ t_inv)

        print(f"Frame {i + 1}: Translation (x,y,z): ({t_total[0, 0]:.3f}, {t_total[1, 0]:.3f}, {t_total[2, 0]:.3f})")

        # Visualize current trajectory
        key = plot_with_estimated_motion(
            ground_truth_positions[:i + 2],
            R_total,
            t_total,
            images,
            K,
            keyframe_idx=last_keyframe_idx,  # Last keyframe
            current_frame_idx=i + 1,         # Current frame
            inlier_pts1=inlier_pts1,
            inlier_pts2=inlier_pts2,
            max_frames=i + 2,
            keyframes=keyframes,
            set=dataset_num
        )

        while i == 0:
            key = cv2.waitKey(0)
            print(f"Key code: {key}")  # Optional: see what code is returned
            if key == 13:  # Enter key
                break

        if key == 27:  # ESC key
            print("ESC pressed. Exiting...")
            break

    print("\nProcessing complete. Press ESC to exit.")
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()