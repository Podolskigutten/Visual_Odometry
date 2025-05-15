import cv2
import numpy as np
import os

def estimate_motion_from_correspondences(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=0.5, prob=0.999)
    points, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    inlier_pts1 = pts1[mask_pose.ravel() == 1]
    inlier_pts2 = pts2[mask_pose.ravel() == 1]
    inliers = inlier_pts1.shape[0]
    print(f"Number of inliers: {inliers}")
    return R, t, inlier_pts1, inlier_pts2

def plot_with_estimated_motion(ground_truth_positions, R_total, t_total, image_list, K, keyframe_idx, current_frame_idx, inlier_pts1=None, inlier_pts2=None, max_frames=1000, keyframes=None, set=None):
    if not hasattr(plot_with_estimated_motion, 'canvas'):
        win_size = 800
        plot_with_estimated_motion.canvas = np.zeros((win_size, win_size, 3), dtype=np.uint8)
        plot_with_estimated_motion.center = win_size // 2
        if set == 0:
            plot_with_estimated_motion.scale = 0.8
        elif set == 1:
            plot_with_estimated_motion.scale = 0.18
        plot_with_estimated_motion.prev_gt = (plot_with_estimated_motion.center, plot_with_estimated_motion.center)
        plot_with_estimated_motion.prev_est = (plot_with_estimated_motion.center, plot_with_estimated_motion.center)
        plot_with_estimated_motion.estimated_path = []
        plot_with_estimated_motion.draw_lines = True  # Initialize toggle state for correspondence lines
        cv2.line(plot_with_estimated_motion.canvas,
                 (plot_with_estimated_motion.center, plot_with_estimated_motion.center),
                 (plot_with_estimated_motion.center + 25, plot_with_estimated_motion.center),
                 (0, 125, 255), 2)
        cv2.line(plot_with_estimated_motion.canvas,
                 (plot_with_estimated_motion.center, plot_with_estimated_motion.center),
                 (plot_with_estimated_motion.center, plot_with_estimated_motion.center - 25),
                 (255, 125, 0), 2)
        cv2.putText(plot_with_estimated_motion.canvas, 'X',
                    (plot_with_estimated_motion.center + 25, plot_with_estimated_motion.center + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 255), 1)
        cv2.putText(plot_with_estimated_motion.canvas, 'Z',
                    (plot_with_estimated_motion.center - 20, plot_with_estimated_motion.center - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 125, 0), 1)

    canvas = plot_with_estimated_motion.canvas
    center = plot_with_estimated_motion.center
    scale = plot_with_estimated_motion.scale
    prev_gt = plot_with_estimated_motion.prev_gt
    display_canvas = canvas.copy()

    if len(ground_truth_positions) > 0:
        gt_x, gt_z = ground_truth_positions[-1]
        gt_draw_x = int(center + gt_x * scale)
        gt_draw_y = int(center - gt_z * scale)
        cv2.line(canvas, prev_gt, (gt_draw_x, gt_draw_y), (0, 255, 0), 2)
        cv2.line(display_canvas, prev_gt, (gt_draw_x, gt_draw_y), (0, 255, 0), 2)
        plot_with_estimated_motion.prev_gt = (gt_draw_x, gt_draw_y)

    est_x = t_total[0, 0]
    est_z = t_total[2, 0]
    plot_with_estimated_motion.estimated_path.append((est_x, est_z))

    prev_est_point = (center, center)
    for point_idx, (x, z) in enumerate(plot_with_estimated_motion.estimated_path):
        est_draw_x = int(center + x * scale)
        est_draw_y = int(center - z * scale)
        cv2.line(canvas, prev_est_point, (est_draw_x, est_draw_y), (0, 0, 255), 2)
        cv2.line(display_canvas, prev_est_point, (est_draw_x, est_draw_y), (0, 0, 255), 2)
        prev_est_point = (est_draw_x, est_draw_y)
        if point_idx >= len(plot_with_estimated_motion.estimated_path) - 5:
            cv2.circle(display_canvas, (est_draw_x, est_draw_y), 3, (0, 100, 255), -1)

        # Optional: Mark keyframes (commented out to match original visualization)
        # if keyframes and point_idx in keyframes:
        #     cv2.circle(display_canvas, (est_draw_x, est_draw_y), 5, (0, 255, 255), 1)

    position_text = f"Pos: ({est_x:.2f}, {est_z:.2f})"
    cv2.putText(display_canvas, position_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    path_length = 0
    for i in range(1, len(plot_with_estimated_motion.estimated_path)):
        x1, z1 = plot_with_estimated_motion.estimated_path[i - 1]
        x2, z2 = plot_with_estimated_motion.estimated_path[i]
        path_length += np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)

    length_text = f"Path Length: {path_length:.2f}"
    cv2.putText(display_canvas, length_text, (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Optional: Add keyframe count and current frame
    if keyframes:
        keyframe_text = f"Keyframes: {len(keyframes)}"
        cv2.putText(display_canvas, keyframe_text, (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if current_frame_idx:
        keyframe_text = f"Current frame: {current_frame_idx}"
        cv2.putText(display_canvas, keyframe_text, (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display feature correspondences (last keyframe on left, current frame on right)
    if 0 <= keyframe_idx < len(image_list) and 0 <= current_frame_idx < len(image_list):
        img1 = image_list[keyframe_idx].copy()      # Last keyframe
        img2 = image_list[current_frame_idx].copy() # Current frame

        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        img_combined = np.hstack((img1, img2))  # Last keyframe (left), current frame (right)
        width = img1.shape[1]

        if inlier_pts1 is not None and inlier_pts2 is not None:
            for pt1, pt2 in zip(inlier_pts1, inlier_pts2):
                pt1 = tuple(np.round(pt1).astype(int))
                pt2 = tuple(np.round(pt2).astype(int) + np.array([width, 0]))
                color = tuple(np.random.randint(0, 255, 3).tolist())
                # Always draw circles
                cv2.circle(img_combined, pt1, 3, color, -1)
                cv2.circle(img_combined, pt2, 3, color, -1)
                # Draw lines only if toggle is enabled
                if plot_with_estimated_motion.draw_lines:
                    cv2.line(img_combined, pt1, pt2, color, 1)

        img_resized = cv2.resize(img_combined, (int(img_combined.shape[1] * 0.5), int(img_combined.shape[0] * 0.5)))
        cv2.imshow('Feature Correspondences', img_resized)

    else:
        print(f"Invalid indices: keyframe_idx={keyframe_idx}, current_frame_idx={current_frame_idx}")

    cv2.imshow('Trajectory', display_canvas)
    key = cv2.waitKey(30)

    # Toggle correspondence lines with 'l' key
    if key == ord('l'):  # ASCII for 'l' is 108
        plot_with_estimated_motion.draw_lines = not plot_with_estimated_motion.draw_lines
        print(f"Correspondence lines {'enabled' if plot_with_estimated_motion.draw_lines else 'disabled'}")

    return key