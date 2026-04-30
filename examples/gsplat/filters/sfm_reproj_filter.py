import numpy as np


def project_points(points_3d, c2w, K):
    """
    Project world points into image using camera intrinsics.

    Args:
        points_3d: (N, 3) world points
        c2w: (4, 4) camera-to-world matrix
        K: (3, 3) intrinsics

    Returns:
        points_proj: (M, 2) projected 2D points
        valid_mask: (N,) mask of points in front of camera
    """

    # Convert to world-to-camera
    w2c = np.linalg.inv(c2w)

    R = w2c[:3, :3]
    t = w2c[:3, 3]  # (3,)

    # World -> camera
    points_cam = (R @ points_3d.T).T + t.reshape(1, 3)

    # Keep points in front of camera
    z = points_cam[:, 2]
    valid_mask = z > 1e-6
    points_cam = points_cam[valid_mask]

    if len(points_cam) == 0:
        return np.zeros((0, 2)), valid_mask

    # Project
    points_proj = (K @ points_cam.T).T
    points_proj = points_proj[:, :2] / points_proj[:, 2:3]

    return points_proj, valid_mask


def sample_image_colors(image, projected_points):
    """
    Sample RGB colors from image at projected pixel locations.

    Args:
        image: H x W x 3 (assumed float32 in [0,1] OR [0,255] consistently)
        projected_points: (M, 2)

    Returns:
        sampled_colors: (K, 3)
        valid_mask: mask of points inside image bounds
    """

    H, W = image.shape[:2]

    x = np.round(projected_points[:, 0]).astype(int)
    y = np.round(projected_points[:, 1]).astype(int)

    valid_mask = (
        (x >= 0) & (x < W) &
        (y >= 0) & (y < H)
    )

    x = x[valid_mask]
    y = y[valid_mask]

    sampled_colors = image[y, x].astype(np.float32)

    return sampled_colors, valid_mask


def compute_photometric_error(image, sfm_points, sfm_rgbs, c2w, K):
    """
    Photometric error:
        mean || rendered_rgb - sfm_rgb ||

    Args:
        image: rendered/refined image
        sfm_points: (N, 3)
        sfm_rgbs: (N, 3)
        c2w: (4, 4)
        K: (3, 3)

    Returns:
        scalar error
    """

    projected_points, front_mask = project_points(
        sfm_points, c2w, K
    )

    sfm_rgbs = sfm_rgbs[front_mask]

    img_colors, img_mask = sample_image_colors(
        image,
        projected_points
    )

    sfm_rgbs = sfm_rgbs[img_mask]

    if len(img_colors) == 0:
        return np.inf

    errors = np.linalg.norm(img_colors - sfm_rgbs, axis=1)
    return np.mean(errors)


def should_discard_refined_photometric(rendered_img, refined_img, sfm_points, sfm_rgbs, c2w, K, verbose=False):
    """
    Discard refined if photometric error is worse than rendered.
    """

    rendered_error = compute_photometric_error(
        rendered_img, sfm_points, sfm_rgbs, c2w, K
    )

    refined_error = compute_photometric_error(
        refined_img, sfm_points, sfm_rgbs, c2w, K
    )

    if verbose:
        print(f"\nRendered error: {rendered_error:.6f}")
        print(f"Refined error:  {refined_error:.6f}")

    if refined_error > rendered_error:
        if verbose:
            print("Discard refined (higher photometric error)")
        return True

    if verbose:
        print("Keep refined")
    return False