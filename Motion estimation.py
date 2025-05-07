import numpy as np
import cv2


# Function to generate motion sequence: move forward N steps, then turn
def generate_motion_sequence(move_steps=30, turns=3, move_distance=0.1, turn_angle_deg=90, curvature_factor=0.3):
    motions = []
    current_angle_deg = 0

    for _ in range(turns):
        # Move forward with increasing or decreasing curvature
        for step in range(move_steps):
            # Simulate sharper curvature by increasing the angle change
            current_angle_deg += curvature_factor * np.sin(step / float(move_steps) * 2 * np.pi)  # Increased curvature
            angle_rad = np.radians(current_angle_deg)
            direction = np.array([  # direction along x, z
                [np.cos(angle_rad)],
                [0],
                [np.sin(angle_rad)]
            ])
            R = np.eye(3)  # No rotation while moving forward
            t = move_distance * direction
            motions.append((R, t))

        # Turn in place with some rotation angle change
        turn_angle_rad = np.radians(turn_angle_deg)
        R = cv2.Rodrigues(np.array([0, turn_angle_rad, 0]))[0]  # rotation matrix for turning
        t = np.zeros((3, 1))  # no translation during turning
        motions.append((R, t))
        current_angle_deg += turn_angle_deg

    return motions


# Function to estimate essential matrix from point correspondences
def estimate_essential_matrix(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask


# Function to recover camera pose (rotation and translation) from the essential matrix
def recover_pose(E, pts1, pts2, K):
    _, R_est, t_est, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R_est, t_est


# Function to update and visualize trajectory in real-time
def trajectory_plot(motions, map_img, origin, win_size=800):
    # Initialize pose
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))
    trajectory = []

    # Loop through each motion and update the trajectory
    for R, t in motions:
        t_total += R_total @ t  # update position
        R_total = R @ R_total  # accumulate rotation
        trajectory.append(t_total.copy())

        # Draw current position on the map
        pos = t_total.flatten()
        x, z = pos[0], pos[2]  # we work in x-z plane
        draw_x = int(origin[0] + x * 50)  # scale by 50 for better visibility
        draw_y = int(origin[1] - z * 50)  # flip z to fit the window's coordinate system

        # **Customize appearance of the plot here**
        color = (0, 0, 255)  # Red color for trajectory points
        thickness = 3
        radius = 3  # Size of the marker

        # Draw the current position as a circle (point)
        cv2.circle(map_img, (draw_x, draw_y), radius, color, thickness)

        # Optionally, draw trajectory path line (if you'd like to show the path)
        if len(trajectory) > 1:
            prev_pos = trajectory[-2].flatten()
            prev_x, prev_z = prev_pos[0], prev_pos[2]
            prev_draw_x = int(origin[0] + prev_x * 50)
            prev_draw_y = int(origin[1] - prev_z * 50)
            cv2.line(map_img, (prev_draw_x, prev_draw_y), (draw_x, draw_y), color, 1)  # Line connecting points

        # Display the map with updated trajectory
        cv2.imshow("Trajectory", map_img)
        key = cv2.waitKey(50)
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()


# Main function that ties everything together
def main(pts1, pts2, K):
    # Generate the motion sequence (this could be based on real motion data)
    motions = generate_motion_sequence(move_steps=1000, turns=4, move_distance=0.1, curvature_factor=0.5)  # Increased curvature factor

    # Create OpenCV window for visualization
    win_size = 800
    map_img = np.ones((win_size, win_size, 3), dtype=np.uint8) * 0  # white background
    origin = np.array([win_size // 2, win_size // 2])

    # Estimate the essential matrix from point correspondences
    E, mask = estimate_essential_matrix(pts1, pts2, K)

    # Recover the pose (rotation and translation)
    R_est, t_est = recover_pose(E, pts1, pts2, K)

    # Update the trajectory in real-time based on motion sequence
    trajectory_plot(motions, map_img, origin, win_size)


# Example usage (replace this with real point correspondences)
if __name__ == "__main__":
    # Fake 2D point correspondences (replace with real data)
    pts1 = np.random.rand(10, 2) * 640  # Random points on the first image
    pts2 = pts1 + np.random.normal(0, 1, pts1.shape)  # Slightly noisy points for the second image

    # Camera calibration matrix
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])

    # Call the main function
    main(pts1, pts2, K)
