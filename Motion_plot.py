import cv2
import numpy as np
import os

# Read ground truth (x, z) from the 3x4 matrix (positions only)
def read_ground_truth_positions(file_path):
    positions = []
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 12:
                try:
                    x = float(values[3])
                    z = float(values[11])
                    positions.append((x, z))
                except ValueError:
                    continue
    return positions

# Estimate Essential matrix and recover relative camera motion
def estimate_motion_from_correspondences(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

# Main visualization loop: draw ground truth and estimated motion
def plot_with_estimated_motion(ground_truth_positions, R, t, image_folder, K, win_size=800, max_frames=1000):
    center = win_size // 2
    scale = 100
    canvas = np.zeros((win_size, win_size, 3), dtype=np.uint8)

    cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)

    prev_gt = (center, center)
    prev_est = (center, center)

    # Initialize estimated position
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    for i, (gt_x, gt_z) in enumerate(ground_truth_positions[:max_frames]):
        # --- Ground truth trajectory (green) ---
        gt_draw_x = int(center + gt_x * scale)
        gt_draw_y = int(center - gt_z * scale)
        cv2.line(canvas, prev_gt, (gt_draw_x, gt_draw_y), (0, 255, 0), 2)
        prev_gt = (gt_draw_x, gt_draw_y)

        # --- Estimated motion (red) ---
        t_total += R_total @ t
        R_total = R @ R_total
        est_x = t_total[0, 0]
        est_z = t_total[2, 0]
        est_draw_x = int(center + est_x * scale)
        est_draw_y = int(center - est_z * scale)
        cv2.line(canvas, prev_est, (est_draw_x, est_draw_y), (0, 0, 255), 2)
        prev_est = (est_draw_x, est_draw_y)

        # --- Display image ---
        img_path = os.path.join(image_folder, f"{i:06d}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, (win_size, win_size))
                cv2.imshow('Image Window', img_resized)

        # Show updated canvas
        cv2.imshow('Trajectory', canvas)
        if cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Load ground truth
    ground_truth_file = 'Images/poses_ground_truth/00.txt'
    image_folder = 'Images/00/image_0/'
    ground_truth_positions = read_ground_truth_positions(ground_truth_file)

    # Dummy correspondences — replace with real matching pixel points
    pts1 = np.array([[100, 150], [200, 120], [300, 300], [400, 310], [250, 230]], dtype=np.float32)
    pts2 = np.array([[102, 152], [202, 122], [302, 298], [398, 308], [252, 232]], dtype=np.float32)

    # Camera intrinsics (example from KITTI)
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])

    # Estimate motion from point correspondences
    #R, t = estimate_motion_from_correspondences(pts1, pts2, K)

    # Plot ground truth and estimated trajectory
    #plot_with_estimated_motion(ground_truth_positions, R, t, image_folder, K, max_frames=1000)
