import numpy as np
import cv2
from PIL import Image

def gs_like_corruption(
    img: Image.Image,
    noise_std=0.03,
    blur_sigma=1.0,
    num_blobs=200,
    blob_radius_range=(3, 15)
) -> Image.Image:
    """
    Apply Gaussian-splatting-like corruption to a PIL image.

    Parameters
    ----------
    img : PIL.Image
        Input image
    noise_std : float
        Standard deviation of Gaussian noise
    blur_sigma : float
        Sigma for Gaussian blur
    num_blobs : int
        Number of translucent blobs
    blob_radius_range : tuple
        Min and max blob radius

    Returns
    -------
    PIL.Image
        Corrupted image
    """

    # Convert PIL -> numpy
    img_np = np.array(img).astype(np.float32) / 255.0
    h, w, c = img_np.shape

    # 1. Gaussian blur
    ksize = int(blur_sigma * 4 + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img_np, (ksize, ksize), blur_sigma)

    # 2. Add Gaussian noise
    noise = np.random.normal(0, noise_std, img_np.shape)
    corrupted = np.clip(blurred + noise, 0, 1)

    # 3. Add translucent Gaussian blobs
    for _ in range(num_blobs):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        radius = np.random.randint(blob_radius_range[0], blob_radius_range[1])

        color = corrupted[y, x]

        mask = np.zeros((h, w), np.float32)
        cv2.circle(mask, (x, y), radius, 1, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), radius / 2)

        corrupted = corrupted * (1 - mask[..., None]) + color * mask[..., None]

    # Convert numpy -> PIL
    corrupted = (corrupted * 255).astype(np.uint8)
    return Image.fromarray(corrupted)