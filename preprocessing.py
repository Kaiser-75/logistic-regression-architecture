import numpy as np
import cv2
from PIL import Image

def preprocess_image(path, size=(30, 30), apply_edges=True):
    img = Image.open(path).convert('L')
    img = img.resize(size)
    img_np = np.array(img).astype(np.float32)

  
    img_np = cv2.GaussianBlur(img_np, (3, 3), sigmaX=0.8)

    if apply_edges:
        sobel_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
        edge_img = np.sqrt(sobel_x**2 + sobel_y**2)
        img_np = (edge_img - edge_img.min()) / (edge_img.max() - edge_img.min() + 1e-8)
    else:
        img_np /= 255.0

    return img_np.flatten()
