# Visual_Odometry
Visual odometry for computer vision course at UiO
![Diagram](./project_plan.png)

# How to:
The feature detection algorithm has been moved to feature_detector.py for offline computation. This script precomputes and saves all keypoints and descriptors, so when the main script is run, feature extraction does not need to be repeated. This significantly reduces runtime, especially on hardware with limited processing power.
