# Test for new file yeeehaaaaa
import cv2 as cv
import os

class ImageLoader:
    def __init__(self, filepath, rate_hz=10):
        self.default_rate = 10  # original rate of images
        self.filepath = filepath
        self.rate_hz = rate_hz
        self.image_files = sorted([
            f for f in os.listdir(filepath)
            if f.lower().endswith(('.png'))
        ])

    def get_indices(self):
        step = int(self.default_rate / self.rate_hz)
        return list(range(0, len(self.image_files), step))

    def load_images(self):
        indices = self.get_indices()
        images = []
        for i in indices:
            full_path = os.path.join(self.filepath, self.image_files[i])
            img = cv.imread(full_path)
            if img is not None:
                images.append(img)
        return images
