import os
import numpy as np
import cv2
from image_processing import ImageLoader, FeatureDetector, FeatureMatcher
from motion_plot import estimate_motion_from_correspondences, plot_with_estimated_motion

def main():
    # Choose dataset
    set = 1 # swap to 1 for set 01
    if set == 0:
        print("Data set 00 chosen")
        # Camera intrinsics (KITTI)
        K = np.array([[718.856, 0, 607.1928],
                    [0, 718.856, 185.2157],
                    [0, 0, 1]])

        # Load images
        path_images = os.path.join("Images", "00", "image_0")
        path_ground_truth = 'Images/poses_ground_truth/00.txt'

    elif set == 1:
        print("Data set 01 chosen")
        # Camera intrinsics (KITTI)
        K = np.array([[718.856, 0, 607.1928],
                    [0, 718.856, 185.2157],
                    [0, 0, 1]])

        # Load images
        path_images = os.path.join("Images", "01", "image_0")
        path_ground_truth = 'Images/poses_ground_truth/01.txt'
    else:
        print("No valid dataset chosen")


    rate = 5
    loader = ImageLoader(path_images, path_ground_truth, desired_rate=rate)
    images, ground_truth_positions = loader.load_images()
    print(f"Loaded {len(images)} images")

    # Choose feature detection method
    method = 'ORB'

    # Detect features in all images
    detector = FeatureDetector(method)
    features = detector.detect_all_features(images)
    print(f"Extracted features from {len(features)} images")

    # Feature matching setup
    matcher = FeatureMatcher(method)

    # Initialize motion tracking variables
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    # Keyframe selection parameters
    min_inliers = 10  # Minimum inliers for reliable motion estimation
    min_translation = 0.1  # Minimum translation norm to consider a keyframe
    last_keyframe_idx = 0
    keyframes = [0]  # Store keyframe indices

    # Scale factor for visualization
    if set == 0:
        initial_scale = 0.75 * (10/rate)
    elif set == 1:
        initial_scale = 2.4 * (10/rate)


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
            set=set
        )

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