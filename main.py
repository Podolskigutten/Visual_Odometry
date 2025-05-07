import os
import numpy as np
import matplotlib as plt
import cv2 as cv
from dataloader import ImageLoader



def main():

    #start by loading images
    path = os.path.join("Images", "00", "image_0")

    loader = ImageLoader(path, 2)
    images = loader.load_images()
    print(f"Loaded {len(images)} images")



if __name__ == "__main__":
    main()