import numpy as np
import cv2
import os


# Function to read ground truth poses from a text file
def read_ground_truth_poses(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.split()))
            if len(values) == 12:  # 3x4 matrix: 3 rotation + 3 translation
                # Construct the pose matrix from the 12 values
                R = np.array(values[:9]).reshape(3, 3)
                t = np.array(values[9:]).reshape(3, 1)
                poses.append((R, t))
    return poses

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
# Function to update and visualize trajectory in real-time
# Function to update and visualize trajectory in real-time, along with updating the same window with images
# Function to update and visualize trajectory in real-time, along with updating the same window with images
def trajectory_plot(motions, ground_truth_poses, map_img, origin, win_size=800, image_folder='Images/00/image_0/'):
    # Initialize pose
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))
    trajectory = []

    # Create a window for the image display
    cv2.namedWindow("Image Window", cv2.WINDOW_NORMAL)  # Window to display the images (resizable)

    # Loop through each motion and update the trajectory
    for i, (R, t) in enumerate(motions):
        scale = 1  # Scale factor for better visibility
        t_total += R_total @ t  # Update position
        R_total = R @ R_total  # Accumulate rotation
        trajectory.append(t_total.copy())

        # Check if the index 'i' is within the length of ground truth poses
        if i < len(ground_truth_poses):
            gt_R, gt_t = ground_truth_poses[i]
            gt_pos = gt_t.flatten()
            x, z = gt_pos[0], gt_pos[2]  # We work in the x-z plane
            draw_x = int(origin[0] + x * scale)  # Scale by 50 for better visibility
            draw_y = int(origin[1] - z * scale)  # Flip z to fit the window's coordinate system

            # Load the image corresponding to the ground truth position
            image_filename = f"{i:06d}.png"  # Assuming images are named 000000.png, 000001.png, ...
            image_path = os.path.join(image_folder, image_filename)

            if os.path.exists(image_path):
                # Read the image
                img = cv2.imread(image_path)
                if img is not None:
                    # Resize image if needed to match the window size
                    img_resized = cv2.resize(img, (win_size, win_size))
                    cv2.imshow("Image Window", img_resized)  # Update the image window
                else:
                    print(f"Warning: Could not read image at {image_path}")
            else:
                print(f"Warning: Image file {image_path} does not exist.")

            # Draw the ground truth position as a circle (point)
            color = (255, 0, 0)  # Blue color for ground truth
            thickness = 3
            radius = 3  # Size of the marker
            cv2.circle(map_img, (draw_x, draw_y), radius, color, thickness)

        # Draw the current motion position on the map
        pos = t_total.flatten()
        x, z = pos[0], pos[2]  # we work in the x-z plane
        draw_x = int(origin[0] + x * scale)  # scale by 50 for better visibility
        draw_y = int(origin[1] - z * scale)  # flip z to fit the window's coordinate system
        color = (0, 0, 255)  # Red color for trajectory points
        thickness = 3
        radius = 3  # Size of the marker
        # Draw the current position as a circle (point)
        cv2.circle(map_img, (draw_x, draw_y), radius, color, thickness)

        # Optionally, draw trajectory path line (if you'd like to show the path)
        if len(trajectory) > 1:
            prev_pos = trajectory[-2].flatten()
            prev_x, prev_z = prev_pos[0], prev_pos[2]
            prev_draw_x = int(origin[0] + prev_x * scale)
            prev_draw_y = int(origin[1] - prev_z * scale)
            cv2.line(map_img, (prev_draw_x, prev_draw_y), (draw_x, draw_y), color, 1)  # Line connecting points

        # Display the map with updated trajectory
        cv2.imshow("Trajectory", map_img)

        # Wait for a short time to update the image and map
        key = cv2.waitKey(50)  # Delay in milliseconds
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()



# Main function that ties everything together
def main(pts1, pts2, K, ground_truth_file):
    # Generate the motion sequence (this could be based on real motion data)
    motions = generate_motion_sequence(move_steps=1000, turns=4, move_distance=0.1, curvature_factor=0.5)  # Increased curvature factor

    # Read the ground truth poses from the file
    ground_truth_poses = read_ground_truth_poses(ground_truth_file)

    # Create OpenCV window for visualization
    win_size = 800
    map_img = np.ones((win_size, win_size, 3), dtype=np.uint8) * 0  # white background
    origin = np.array([win_size // 2, win_size // 2])  # Define the origin point in the center of the image

    # Estimate the essential matrix from point correspondences (dummy values in this case)
    E, mask = estimate_essential_matrix(pts1, pts2, K)

    # Recover the pose (rotation and translation) (this part can be replaced with real data)
    R_est, t_est = recover_pose(E, pts1, pts2, K)

    # Update the trajectory in real-time based on motion sequence and ground truth
    trajectory_plot(motions, ground_truth_poses, map_img, origin, win_size=800, image_folder='Images/00/image_0/')


# Example usage (replace this with real point correspondences)
if __name__ == "__main__":
    # Fake 2D point correspondences (replace with real data)
    pts1 = np.random.rand(10, 2) * 640  # Random points on the first image
    pts2 = pts1 + np.random.normal(0, 1, pts1.shape)  # Slightly noisy points for the second image

    # Camera calibration matrix (example, should be replaced with real data)
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])

    # Path to ground truth poses file
    ground_truth_file = "Images/poses_ground_truth/00.txt"

    # Call the main function
    main(pts1, pts2, K, ground_truth_file)