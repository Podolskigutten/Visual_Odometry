import os
import numpy as np
import matplotlib as plt
import cv2 
from image_processing import ImageLoader, FeatureDetecor, FeatureMatcher



def main():

    #start by loading images
    path = os.path.join("Images", "00", "image_0")

    loader = ImageLoader(path, 2)
    images = loader.load_images()
    print(f"Loaded {len(images)} images")

    # Choose feature detection method
    method = 'SIFT'  # Choose between SIFT, ORB and AKAZE

    # Detect features in all images, results in keypoints and descriptors
    detector = FeatureDetecor(method) # Choose between SIFT, ORB and AKAZE
    features = detector.detect_all_features(images)

    # Pairwise match features, esitmate essential matrix 
    # and calculate the translation between image pairs
    matcher = FeatureMatcher(method)

    # Process consecutive image pairs
    for i in range(len(features) - 1):
        # Get current pair of features
        kp1, des1 = features[i]
        kp2, des2 = features[i + 1]
        
        # Match features between consecutive images
        coords1, coords2 = matcher.match_features(kp1, des1, kp2, des2)

        # Now comes your code broooooo, estimate the shit out of the Essential matrix

if __name__ == "__main__":
    main()