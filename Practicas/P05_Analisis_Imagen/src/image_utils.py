from pathlib import Path
import numpy as np
from skimage import io, color, filters, exposure, morphology, measure
import cv2

def load_gray(path: Path) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:
        return color.rgb2gray(img)
    return img.astype(float)/255.0
