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
    points, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    inlier_pts1 = pts1[mask_pose.ravel() == 1]
    inlier_pts2 = pts2[mask_pose.ravel() == 1]
    return R, t, inlier_pts1, inlier_pts2

    # Count inliers
    inliers = inlier_pts1.shape[0]
    print(f"Number of inliers: {inliers}")

    return R, t, inlier_pts1, inlier_pts2



# Main visualization loop: draw ground truth and estimated motion
def plot_with_estimated_motion(ground_truth_positions, R_total, t_total, image_list, K, frame_index, inlier_pts1=None, inlier_pts2=None, max_frames=1000):


    # Create persistent static storage for trajectory history
    if not hasattr(plot_with_estimated_motion, 'canvas'):
        win_size = 800
        plot_with_estimated_motion.canvas = np.zeros((win_size, win_size, 3), dtype=np.uint8)
        plot_with_estimated_motion.center = win_size // 2
        plot_with_estimated_motion.scale = 0.8  # Adjust this value based on your data scale
        plot_with_estimated_motion.prev_gt = (plot_with_estimated_motion.center, plot_with_estimated_motion.center)
        plot_with_estimated_motion.prev_est = (plot_with_estimated_motion.center, plot_with_estimated_motion.center)
        plot_with_estimated_motion.estimated_path = []  # Store the estimated path points

        # Draw coordinate axes for reference
        cv2.line(plot_with_estimated_motion.canvas,
                 (plot_with_estimated_motion.center, plot_with_estimated_motion.center),
                 (plot_with_estimated_motion.center + 100, plot_with_estimated_motion.center),
                 (0, 125, 255), 1)  # X-axis (amber)
        cv2.line(plot_with_estimated_motion.canvas,
                 (plot_with_estimated_motion.center, plot_with_estimated_motion.center),
                 (plot_with_estimated_motion.center, plot_with_estimated_motion.center - 100),
                 (255, 125, 0), 1)  # Z-axis (blue)
        cv2.putText(plot_with_estimated_motion.canvas, 'X',
                    (plot_with_estimated_motion.center + 110, plot_with_estimated_motion.center + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 255), 1)
        cv2.putText(plot_with_estimated_motion.canvas, 'Z',
                    (plot_with_estimated_motion.center - 15, plot_with_estimated_motion.center - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 125, 0), 1)

    # Use references for better readability
    canvas = plot_with_estimated_motion.canvas
    center = plot_with_estimated_motion.center
    scale = plot_with_estimated_motion.scale
    prev_gt = plot_with_estimated_motion.prev_gt

    # Make a copy of the canvas for this frame
    display_canvas = canvas.copy()

    # Draw ground truth for new point
    if len(ground_truth_positions) > 0:
        gt_x, gt_z = ground_truth_positions[-1]
        gt_draw_x = int(center + gt_x * scale)
        gt_draw_y = int(center - gt_z * scale)  # Negative because y-axis is inverted in image coordinates
        cv2.line(canvas, prev_gt, (gt_draw_x, gt_draw_y), (0, 255, 0), 2)  # Update master canvas
        cv2.line(display_canvas, prev_gt, (gt_draw_x, gt_draw_y), (0, 255, 0), 2)  # Update display canvas
        plot_with_estimated_motion.prev_gt = (gt_draw_x, gt_draw_y)

    # For the estimated path, we need to invert the sign for the translation
    # to match the ground truth direction (camera vs. world coordinate system)
    est_x = t_total[0, 0]
    est_z = t_total[2, 0]

    # Store the current estimated position
    plot_with_estimated_motion.estimated_path.append((est_x, est_z))

    # Draw the entire estimated path
    prev_est_point = (center, center)  # Start from center
    for point_idx, (x, z) in enumerate(plot_with_estimated_motion.estimated_path):
        # Draw estimated position point
        est_draw_x = int(center + x * scale)
        est_draw_y = int(center - z * scale)  # Negative because y-axis is inverted

        # Draw line segment from previous point
        cv2.line(canvas, prev_est_point, (est_draw_x, est_draw_y), (0, 0, 255), 2)
        cv2.line(display_canvas, prev_est_point, (est_draw_x, est_draw_y), (0, 0, 255), 2)

        # Update previous point
        prev_est_point = (est_draw_x, est_draw_y)

        # Draw point markers for better visibility (only for a few recent points)
        if point_idx >= len(plot_with_estimated_motion.estimated_path) - 5:
            cv2.circle(display_canvas, (est_draw_x, est_draw_y), 3, (0, 100, 255), -1)

    # Add text for current position
    position_text = f"Pos: ({est_x:.2f}, {est_z:.2f})"
    cv2.putText(display_canvas, position_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add path length
    path_length = 0
    for i in range(1, len(plot_with_estimated_motion.estimated_path)):
        x1, z1 = plot_with_estimated_motion.estimated_path[i - 1]
        x2, z2 = plot_with_estimated_motion.estimated_path[i]
        path_length += np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)

    length_text = f"Path Length: {path_length:.2f}"
    cv2.putText(display_canvas, length_text, (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if 0 <= frame_index < len(image_list):
        img = image_list[frame_index].copy()  # COPY HERE

        # Draw inlier points if available
        if inlier_pts2 is not None and len(inlier_pts2) > 0:
            inlier_pts2 = inlier_pts2.reshape(-1, 2)

            for pt in inlier_pts2:
                pt_int = tuple(np.round(pt).astype(int))
                cv2.circle(img, pt_int, 4, (0, 255, 0), 1)  # Bright green

        img_resized = cv2.resize(img, (int(1920 / 3), int(1080 / 3)))
        cv2.imshow('Image Window', img_resized)
        cv2.waitKey(1)
    else:
        print(f"No image available at index {frame_index}")


    # Show updated canvas
    cv2.imshow('Trajectory', display_canvas)
    key = cv2.waitKey(30)

    return key