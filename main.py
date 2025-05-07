import os
import numpy as np
import matplotlib as plt
import cv2 
from image_processing import ImageLoader, FeatureDetecor



def main():

    #start by loading images
    path = os.path.join("Images", "00", "image_0")

    loader = ImageLoader(path, 2)
    images = loader.load_images()
    print(f"Loaded {len(images)} images")

    detector = FeatureDetecor('SIFT') # Choose between SIFT, ORB and AKAZE
    features = detector.detect_all_features(images)


if __name__ == "__main__":
    main()