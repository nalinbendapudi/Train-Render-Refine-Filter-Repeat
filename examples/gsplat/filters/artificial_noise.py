import numpy as np
import cv2
from PIL import Image

def gs_like_corruption(
    img: Image.Image,
    rng = np.random.default_rng(42),
    noise_std=0.03,
    blur_sigma_ratio=0.02,
    num_blobs=200,
    blob_radius_range_ratio=(0.05, 0.1)
) -> Image.Image:
    """
    Apply Gaussian-splatting-like corruption to a PIL image.

    Parameters
    ----------
    img : PIL.Image
        Input image
    noise_std : float
        Standard deviation of Gaussian noise
    blur_sigma_ratio : float
        Sigma for Gaussian blur as a ratio of the image diagonal
    num_blobs : int
        Number of translucent blobs
    blob_radius_range_ratio : tuple
        Min and max blob radius as a ratio of the image diagonal

    Returns
    -------
    PIL.Image
        Corrupted image
    """

    # Convert PIL -> numpy
    img_np = np.array(img).astype(np.float32) / 255.0
    h, w, _ = img_np.shape
    diag = np.sqrt(h**2 + w**2)

    # 1. Gaussian blur
    blur_sigma = blur_sigma_ratio * diag
    ksize = int(blur_sigma * 4 + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img_np, (ksize, ksize), blur_sigma)

    # 2. Add Gaussian noise
    noise = rng.normal(0, noise_std, img_np.shape)
    corrupted = np.clip(blurred + noise, 0, 1)

    # 3. Add translucent Gaussian blobs
    rmin = int(blob_radius_range_ratio[0] * diag)
    rmax = int(blob_radius_range_ratio[1] * diag)
    
    for _ in range(num_blobs):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        radius = rng.integers(rmin, rmax)

        color = corrupted[y, x]

       # ---- LOCAL PATCH ONLY ----
        x0 = max(0, x - radius)
        x1 = min(w, x + radius)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius)

        patch_h = y1 - y0
        patch_w = x1 - x0

        # Create small mask
        mask = np.zeros((patch_h, patch_w), np.float32)

        cv2.circle(mask, (x - x0, y - y0), radius, 1, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), radius / 2)

        patch = corrupted[y0:y1, x0:x1]

        # Blend locally
        corrupted[y0:y1, x0:x1] = (
            patch * (1 - mask[..., None]) +
            color * mask[..., None]
        )

    # Convert numpy -> PIL
    corrupted = (corrupted * 255).astype(np.uint8)
    return Image.fromarray(corrupted)