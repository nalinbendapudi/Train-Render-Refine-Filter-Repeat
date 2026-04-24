import torch
import numpy as np
import cv2
import os


def backproject_depth(depth, rgb, K, c2w, sample_stride=8):
    """
    depth: H x W
    rgb:   H x W x 3
    K:     3 x 3
    c2w:   4 x 4 camera-to-world pose

    returns:
        points_world: [N, 3]
        colors:       [N, 3]
    """

    H, W = depth.shape

    # subsample pixels
    ys, xs = np.meshgrid(
        np.arange(0, H, sample_stride),
        np.arange(0, W, sample_stride),
        indexing="ij"
    )

    zs = depth[ys, xs]  # [N]
    colors = rgb[ys, xs] / 255.0

    valid = zs > 0
    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]
    colors = colors[valid]

    pixels = np.stack([
        xs,
        ys,
        np.ones_like(xs)
    ], axis=0)  # [3, N]

    K_inv = np.linalg.inv(K)

    # camera coordinates
    points_cam = K_inv @ pixels
    points_cam = points_cam * zs[None, :]

    # homogeneous
    ones = np.ones((1, points_cam.shape[1]))
    points_cam_h = np.concatenate([points_cam, ones], axis=0)

    # world coordinates
    points_world = (c2w @ points_cam_h)[:3].T

    return torch.from_numpy(points_world).float(), \
           torch.from_numpy(colors).float()

def init_from_mono_depth(
    parser,
    trainset,
    sample_stride=8,
):
    all_points = []
    all_rgbs = []

    depth_dir = os.path.join(parser.data_dir, f"depth_{parser.factor}")
    
    for idx in trainset.indices:    
        img_path = parser.image_paths[idx]
        img_name = os.path.basename(img_path)
        depth_path = os.path.join(depth_dir, img_name.replace(".JPG", ".png"))
        K = parser.Ks_dict[parser.camera_ids[idx]]
        c2w = parser.camtoworlds[idx]

        rgb = cv2.imread(img_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        pts, cols = backproject_depth(
            depth,
            rgb,
            K,
            c2w,
            sample_stride
        )

        all_points.append(pts)
        all_rgbs.append(cols)

    points = torch.cat(all_points, dim=0)
    rgbs = torch.cat(all_rgbs, dim=0)

    return points, rgbs