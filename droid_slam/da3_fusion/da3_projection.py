"""
DA3-Specific Projection and Filtering Operations

Python implementation of inverse projection and depth filtering specifically
designed for DA3 depth maps (metric depth, not disparity).

Key Differences from DROID backends:
    1. Uses depth directly (not inverse depth/disparity)
    2. Handles scale-aligned DA3 depth with confidence weighting
    3. Pure Python/PyTorch implementation for flexibility
    4. Optimized for reasonable resolution (240x320 or higher)

Professor-Level Design:
    - Geometric multi-view consistency checking
    - Confidence-weighted filtering
    - Efficient batched operations for real-time performance
"""

import torch
import torch.nn.functional as F
from lietorch import SE3


def iproj_depth(poses, depths, intrinsics, return_local=False):
    """
    Inverse projection from depth maps to 3D points (DA3 version).

    Unlike DROID's disparity-based iproj, this operates on metric depth directly.

    Args:
        poses: Camera poses [N, 7] (tx, ty, tz, qw, qx, qy, qz) - world frame
        depths: Depth maps [N, H, W] in meters
        intrinsics: Camera intrinsics [4] (fx, fy, cx, cy)
        return_local: If True, return points in camera frame instead of world frame

    Returns:
        points_3d: 3D points [N, H, W, 3] in world frame (or camera frame if return_local=True)
    """
    N, H, W = depths.shape
    device = depths.device

    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    # Create pixel grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Back-project to camera frame
    # X = (x - cx) * Z / fx
    # Y = (y - cy) * Z / fy
    # Z = depth
    X_cam = (x_grid - cx) * depths / fx  # [N, H, W]
    Y_cam = (y_grid - cy) * depths / fy
    Z_cam = depths

    # Stack to [N, H, W, 3]
    points_cam = torch.stack([X_cam, Y_cam, Z_cam], dim=-1)

    if return_local:
        return points_cam

    # Transform to world frame
    # Reshape for batch processing: [N*H*W, 3]
    points_cam_flat = points_cam.reshape(N, -1, 3)

    # Homogeneous coordinates
    ones = torch.ones(N, H * W, 1, device=device)
    points_hom = torch.cat([points_cam_flat, ones], dim=2)  # [N, H*W, 4]

    # Get transformation matrices
    T = SE3(poses)  # [N, 7] -> SE3 object
    T_mat = T.matrix()  # [N, 4, 4]

    # Transform: p_world = T @ p_cam
    # [N, 4, 4] @ [N, 4, H*W] -> [N, 4, H*W]
    points_world_hom = torch.bmm(T_mat, points_hom.transpose(1, 2))  # [N, 4, H*W]
    points_world = points_world_hom[:, :3, :].transpose(1, 2)  # [N, H*W, 3]

    # Reshape back to [N, H, W, 3]
    points_world = points_world.reshape(N, H, W, 3)

    return points_world


def depth_filter_multiview(poses, depths, confidences, intrinsics, index,
                           reproj_thresh=0.05, conf_thresh=0.5, min_views=2):
    """
    Multi-view geometric consistency filter for DA3 depth maps.

    Validates depth estimates by checking reprojection consistency across views.
    This is a Python replacement for droid_backends.depth_filter.

    Args:
        poses: Camera poses [N, 7]
        depths: Depth maps [N, H, W]
        confidences: DA3 confidence maps [N, H, W]
        intrinsics: Camera intrinsics [4]
        index: Frame indices to process [M]
        reproj_thresh: Reprojection error threshold (relative to depth)
        conf_thresh: Minimum confidence threshold
        min_views: Minimum number of consistent views required

    Returns:
        consistency_counts: Number of consistent views for each pixel [N, H, W]
    """
    N, H, W = depths.shape
    device = depths.device

    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    # Initialize consistency counts
    counts = torch.zeros(N, H, W, device=device, dtype=torch.int32)

    # Only process frames in index
    frames_to_check = index.cpu().numpy()

    # Pixel grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    for i in frames_to_check:
        if i >= N:
            continue

        # Get depth and confidence for frame i
        depth_i = depths[i]
        conf_i = confidences[i]

        # Skip low-confidence pixels
        valid_mask = (conf_i > conf_thresh) & (depth_i > 0.1) & (depth_i < 100.0)

        if valid_mask.sum() == 0:
            continue

        # Back-project to camera frame
        X_cam_i = (x_grid - cx) * depth_i / fx
        Y_cam_i = (y_grid - cy) * depth_i / fy
        Z_cam_i = depth_i

        # Transform to world frame
        points_cam_i = torch.stack([X_cam_i, Y_cam_i, Z_cam_i, torch.ones_like(depth_i)], dim=-1)  # [H, W, 4]

        T_i = SE3(poses[i:i+1]).matrix()[0]  # [4, 4]
        points_world_i = (T_i @ points_cam_i.reshape(-1, 4).T).T.reshape(H, W, 4)[:, :, :3]  # [H, W, 3]

        # Check consistency with neighboring frames
        for j in range(max(0, i - 5), min(N, i + 6)):
            if j == i:
                continue

            # Transform to frame j
            T_j = SE3(poses[j:j+1]).matrix()[0]
            T_j_inv = torch.linalg.inv(T_j)

            points_j_hom = torch.cat([points_world_i, torch.ones(H, W, 1, device=device)], dim=-1)
            points_cam_j = (T_j_inv @ points_j_hom.reshape(-1, 4).T).T.reshape(H, W, 4)[:, :, :3]

            X_j, Y_j, Z_j = points_cam_j[:, :, 0], points_cam_j[:, :, 1], points_cam_j[:, :, 2]

            # Project to image plane
            u_j = fx * X_j / (Z_j + 1e-8) + cx
            v_j = fy * Y_j / (Z_j + 1e-8) + cy

            # Check if in bounds
            in_bounds = (u_j >= 0) & (u_j < W - 1) & (v_j >= 0) & (v_j < H - 1) & (Z_j > 0.1)

            # Sample depth at projected location
            u_norm = (u_j / (W - 1)) * 2 - 1
            v_norm = (v_j / (H - 1)) * 2 - 1
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]

            depth_j_sampled = F.grid_sample(
                depths[j:j+1].unsqueeze(0),  # [1, 1, H, W]
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            ).squeeze()  # [H, W]

            # Check depth consistency
            depth_error = torch.abs(Z_j - depth_j_sampled) / (Z_j + 1e-8)
            consistent = in_bounds & (depth_error < reproj_thresh) & valid_mask

            counts[i] += consistent.int()

    return counts


def create_point_cloud_from_da3(poses, depths, confidences, intrinsics, images,
                                 scale=1.0, shift=0.0, min_conf=0.5, min_views=2,
                                 resolution_scale=1.0):
    """
    Create colored point cloud from DA3 depth maps with scale alignment.

    This is the main function for visualization, replacing DROID's disparity-based approach.

    Args:
        poses: Camera poses [N, 7]
        depths: DA3 depth maps [N, H, W] (affine-invariant)
        confidences: DA3 confidence maps [N, H, W]
        intrinsics: Camera intrinsics [4]
        images: RGB images [N, 3, H, W]
        scale: Scale factor for depth alignment
        shift: Shift factor for depth alignment
        min_conf: Minimum confidence threshold
        min_views: Minimum consistent views for filtering
        resolution_scale: Downsample factor for performance (0.5 = half res)

    Returns:
        points: Valid 3D points [M, 3]
        colors: RGB colors [M, 3]
    """
    N, H_full, W_full = depths.shape
    device = depths.device

    # Downsample for performance if needed
    if resolution_scale < 1.0:
        H_target = int(H_full * resolution_scale)
        W_target = int(W_full * resolution_scale)

        depths = F.interpolate(
            depths.unsqueeze(1),  # [N, 1, H, W]
            size=(H_target, W_target),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        confidences = F.interpolate(
            confidences.unsqueeze(1),
            size=(H_target, W_target),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        # Adjust intrinsics
        intrinsics = intrinsics.clone()
        intrinsics[0] *= resolution_scale  # fx
        intrinsics[1] *= resolution_scale  # fy
        intrinsics[2] *= resolution_scale  # cx
        intrinsics[3] *= resolution_scale  # cy

    # Apply scale-shift alignment
    depths_aligned = scale * depths + shift
    depths_aligned = torch.clamp(depths_aligned, min=0.2, max=100.0)

    # Multi-view filtering
    index = torch.arange(N, device=device)
    counts = depth_filter_multiview(
        poses, depths_aligned, confidences, intrinsics, index,
        reproj_thresh=0.05, conf_thresh=min_conf, min_views=min_views
    )

    # Valid mask
    valid_mask = (counts >= min_views) & (confidences > min_conf) & (depths_aligned > 0.2)

    # Inverse projection
    points_world = iproj_depth(poses, depths_aligned, intrinsics, return_local=False)  # [N, H, W, 3]

    # Extract colors
    if resolution_scale < 1.0:
        H, W = depths.shape[1], depths.shape[2]
        images_resized = F.interpolate(
            images.float(),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
    else:
        images_resized = images.float()

    # [N, 3, H, W] -> [N, H, W, 3]
    colors = images_resized.permute(0, 2, 3, 1) / 255.0

    # Flatten and filter
    points_flat = points_world.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)
    valid_flat = valid_mask.reshape(-1)

    points_valid = points_flat[valid_flat]
    colors_valid = colors_flat[valid_flat]

    return points_valid, colors_valid


def estimate_scale_from_poses_and_depths(poses_droid, depths_da3, intrinsics,
                                         baseline_thresh=0.05):
    """
    Estimate scale factor by comparing DA3 depth with DROID pose baselines (Professor-level).

    Key insight: For frames with sufficient baseline, the translation magnitude should
    correlate with the scene depth. We use multi-view geometric constraints.

    Args:
        poses_droid: DROID poses [N, 7]
        depths_da3: DA3 depth maps [N, H, W]
        intrinsics: Camera intrinsics [4] (at 1/8 resolution)
        baseline_thresh: Minimum baseline between frames

    Returns:
        scale: Estimated scale factor
        confidence: Confidence score
    """
    N = poses_droid.shape[0]
    device = poses_droid.device

    if N < 3:
        return 1.0, 0.0

    # Find frames with valid DA3 depth (non-zero)
    valid_frames = []
    for i in range(N):
        if depths_da3[i].abs().sum() > 0:
            valid_frames.append(i)

    if len(valid_frames) < 2:
        return 1.0, 0.0

    # Compute baselines between consecutive frames
    poses_mat = SE3(poses_droid).matrix()  # [N, 4, 4]
    translations = poses_mat[:, :3, 3]  # [N, 3]

    scale_estimates = []

    # Only use pairs where BOTH frames have DA3 depth
    for idx_i in range(len(valid_frames)):
        for idx_j in range(idx_i + 1, len(valid_frames)):
            i = valid_frames[idx_i]
            j = valid_frames[idx_j]

            # Compute baseline
            baseline = torch.norm(translations[j] - translations[i]).item()

            if baseline < baseline_thresh:
                continue

            # Get median depth from DA3 (only non-zero values)
            depth_i_valid = depths_da3[i][depths_da3[i] > 0.1]
            depth_j_valid = depths_da3[j][depths_da3[j] > 0.1]

            if len(depth_i_valid) < 100 or len(depth_j_valid) < 100:
                continue  # Not enough valid pixels

            depth_i_median = depth_i_valid.median().item()
            depth_j_median = depth_j_valid.median().item()
            depth_avg = (depth_i_median + depth_j_median) / 2

            if depth_avg < 0.1:
                continue

            # Scale estimation: baseline / average_DA3_depth
            # This assumes DA3 depth is in arbitrary units and needs scaling to match metric poses
            scale_est = baseline / depth_avg

            # Sanity check: scale should be reasonable (0.01 to 100)
            if 0.01 < scale_est < 100.0:
                scale_estimates.append(scale_est)

    if len(scale_estimates) == 0:
        return 1.0, 0.0

    # Robust median with outlier rejection
    scale_tensor = torch.tensor(scale_estimates, device=device)
    scale_median = scale_tensor.median().item()

    # Filter outliers (within 3x of median)
    inliers = scale_tensor[(scale_tensor > scale_median / 3.0) & (scale_tensor < scale_median * 3.0)]

    if len(inliers) > 0:
        scale = inliers.mean().item()  # Mean of inliers for stability
    else:
        scale = scale_median

    confidence = min(len(inliers) / 5.0, 1.0)  # More inliers = higher confidence

    return scale, confidence
