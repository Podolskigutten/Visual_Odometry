import cv2
import numpy as np
import os

# Read x, z positions from the ground truth file (3x4 matrices)
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

# Plot ground truth trajectory and show corresponding images
def plot_trajectory_with_images(positions, image_folder, win_size=800, max_frames=1000):
    center = win_size // 2
    scale = 1
    canvas = np.zeros((win_size, win_size, 3), dtype=np.uint8)

    cv2.namedWindow('Movement', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)

    prev_pos = (center, center)

    for i, (x, z) in enumerate(positions[:max_frames]):
        new_x = int(center + x * scale)
        new_y = int(center - z * scale)

        cv2.line(canvas, prev_pos, (new_x, new_y), (0, 255, 0), 8)
        prev_pos = (new_x, new_y)

        # Load and display corresponding image
        img_path = os.path.join(image_folder, f"{i:06d}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, (win_size, win_size))
                cv2.imshow('Image Window', img_resized)
            else:
                print(f"Warning: Could not read image {img_path}")
        else:
            print(f"Warning: Missing image {img_path}")

        cv2.imshow('Movement', canvas)
        key = cv2.waitKey(30)
        if key == 27:
            break

    cv2.destroyAllWindows()

# Main runner
if __name__ == "__main__":
    ground_truth_file = 'Images/poses_ground_truth/00.txt'
    image_folder = 'Images/00/image_0/'

    positions = read_ground_truth_positions(ground_truth_file)
    plot_trajectory_with_images(positions, image_folder, max_frames=1000)
