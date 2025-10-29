import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def write_bytes_to_file(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)

def read_image_array(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr_from_mse(mse_value):
    if mse_value == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse_value))

def ssim_safe(a, b):
    try:
        return float(ssim(a, b, channel_axis=2))
    except ValueError:
        return 0.0
