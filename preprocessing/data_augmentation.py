## imports 
import albumentations as A
from PIL import Image   
import numpy as np
from preprocessing.config import IMG_SIZE

class Augment:
    def __init__(self, image_size = IMG_SIZE):
        self.transform = A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=10,p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4)
        ])
    def __call__(self, image: Image.Image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        return Image.fromarray(augmented["image"])