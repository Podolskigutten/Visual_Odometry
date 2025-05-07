import numpy as np
import cv2

# Simulate camera intrinsics
K = np.array([[800, 0, 320],
              [0, 800, 240],
              [0,   0,   1]])

# Generate random 3D points in front of the camera
num_points = 100
pts_3d = np.random.uniform(-1, 1, (num_points, 3))
pts_3d[:, 2] += 5  # make sure points are in front of the camera

# Define camera poses (identity and a small motion)
R1 = np.eye(3)
t1 = np.zeros((3, 1))

# Simulate a small translation along X and rotation around Y
angle = np.deg2rad(5)
R2 = cv2.Rodrigues(np.array([0, angle, 0]))[0]
t2 = np.array([[0.1], [0], [0]])

# Project 3D points to image plane (pts1, pts2)
def project_points(P, pts_3d):
    proj = P @ np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1)))).T
    proj = proj[:2] / proj[2]
    return proj.T

P1 = K @ np.hstack((R1, t1))
P2 = K @ np.hstack((R2, t2))

pts1 = project_points(P1, pts_3d)
pts2 = project_points(P2, pts_3d)

# Add a little noise
pts1 += np.random.normal(0, 0.5, pts1.shape)
pts2 += np.random.normal(0, 0.5, pts2.shape)

# Test essential matrix estimation and pose recovery
E, mask = cv2.findEssentialMat(pts1, pts2, K)
_, R_est, t_est, _ = cv2.recoverPose(E, pts1, pts2, K)

print("Estimated Rotation:\n", R_est)
print("Estimated Translation (direction):\n", t_est)

