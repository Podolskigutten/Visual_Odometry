import os
import numpy as np
import matplotlib as plt
import cv2 as cv
from dataloader import ImageLoader

# Load data from working directory
path = os.path.join("Images", "00", "image_0")

loader = ImageLoader(path, 1)
images = loader.load_images()

print(f"Loaded {len(images)} images")
