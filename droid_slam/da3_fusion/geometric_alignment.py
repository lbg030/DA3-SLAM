"""
Geometric Scale-Shift Alignment with Triangulation Validation

This module implements professor-level scale estimation between DA3 (affine-invariant)
and DROID (metric) depth maps using multi-view geometric validation.

Key Insight:
    DA3 produces affine-invariant depth: d_da3 = a * d_true + b
    We need to estimate both scale 's' and shift 't' such that:
    d_metric = s * d_da3 + t

Approach:
    1. Triangulate 3D points from multiple views
    2. Align point clouds via Umeyama algorithm (scale + rotation + translation)
    3. Validate via reprojection error
    4. Robust estimation with RANSAC-like consensus
"""

import torch
import torch.nn.functional as F
import numpy as np
from lietorch import SE3


def triangulate_points(depth_i, pose_i, pose_j, intrinsics, min_parallax=0.01):
    """
    Triangulate 3D points from depth in frame i and poses.

    Args:
        depth_i: Depth map in frame i [H, W]
        pose_i: SE3 pose of frame i [7] (tx, ty, tz, qw, qx, qy, qz)
        pose_j: SE3 pose of frame j [7]
        intrinsics: Camera intrinsics [4] (fx, fy, cx, cy)
        min_parallax: Minimum parallax angle in radians for valid triangulation

    Returns:
        points_3d: Triangulated 3D points in world frame [N, 3]
        valid_mask: Boolean mask of valid points [H, W]
    """
    H, W = depth_i.shape
    device = depth_i.device

    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Back-project to 3D in camera i frame
    X_cam_i = (x - cx) * depth_i / fx
    Y_cam_i = (y - cy) * depth_i / fy
    Z_cam_i = depth_i

    # Points in camera i frame [H, W, 3]
    points_cam_i = torch.stack([X_cam_i, Y_cam_i, Z_cam_i], dim=-1)

    # Transform to world frame
    T_i = SE3(pose_i.unsqueeze(0))  # [1, 7]
    T_i_mat = T_i.matrix()[0]  # [4, 4]

    # Homogeneous coordinates
    points_cam_i_flat = points_cam_i.reshape(-1, 3)  # [H*W, 3]
    ones = torch.ones(points_cam_i_flat.shape[0], 1, device=device)
    points_hom = torch.cat([points_cam_i_flat, ones], dim=1)  # [H*W, 4]

    # Transform to world
    points_world_hom = (T_i_mat @ points_hom.T).T  # [H*W, 4]
    points_world = points_world_hom[:, :3]  # [H*W, 3]

    # Transform to camera j frame
    T_j = SE3(pose_j.unsqueeze(0))
    T_j_mat = T_j.matrix()[0]
    T_j_inv = torch.linalg.inv(T_j_mat)

    points_cam_j_hom = (T_j_inv @ points_world_hom.T).T
    points_cam_j = points_cam_j_hom[:, :3]

    # Compute parallax angle
    # angle = arccos(ray_i · ray_j / (||ray_i|| ||ray_j||))
    ray_i = points_cam_i_flat / (points_cam_i_flat.norm(dim=1, keepdim=True) + 1e-8)
    ray_j = points_cam_j / (points_cam_j.norm(dim=1, keepdim=True) + 1e-8)
    cos_parallax = (ray_i * ray_j).sum(dim=1)
    parallax = torch.acos(torch.clamp(cos_parallax, -1.0, 1.0))

    # Valid points: sufficient parallax and positive depth in both frames
    valid = (
        (parallax > min_parallax) &
        (Z_cam_i.reshape(-1) > 0.1) &
        (points_cam_j[:, 2] > 0.1)
    )

    valid_mask = valid.reshape(H, W)

    return points_world.reshape(H, W, 3), valid_mask


def umeyama_alignment(src_points, dst_points):
    """
    Compute scale, rotation, and translation to align src to dst.

    Uses Umeyama algorithm for 3D-3D alignment with scale.

    Args:
        src_points: Source 3D points [N, 3]
        dst_points: Destination 3D points [N, 3]

    Returns:
        scale: Scale factor (float)
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
    """
    # Center the point clouds
    src_mean = src_points.mean(dim=0)
    dst_mean = dst_points.mean(dim=0)

    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    # Compute scale
    src_scale = torch.sqrt((src_centered ** 2).sum() / len(src_points))
    dst_scale = torch.sqrt((dst_centered ** 2).sum() / len(dst_points))
    scale = dst_scale / (src_scale + 1e-8)

    # Normalize
    src_normalized = src_centered / (src_scale + 1e-8)
    dst_normalized = dst_centered / (dst_scale + 1e-8)

    # Compute rotation via SVD
    H = src_normalized.T @ dst_normalized  # [3, 3]
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.T

    R = V @ U.T

    # Handle reflection case
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    # Compute translation
    t = dst_mean - scale * (R @ src_mean)

    return scale, R, t


def compute_reprojection_error(points_3d, depth_target, pose_target, intrinsics):
    """
    Compute reprojection error of 3D points into target frame.

    Args:
        points_3d: 3D points in world frame [N, 3]
        depth_target: Target depth map [H, W]
        pose_target: Target pose [7]
        intrinsics: Camera intrinsics [4]

    Returns:
        error: Mean reprojection error (float)
        inlier_mask: Inliers with error < threshold [N]
    """
    device = points_3d.device
    H, W = depth_target.shape

    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    # Transform to target camera frame
    T_target = SE3(pose_target.unsqueeze(0))
    T_target_mat = T_target.matrix()[0]
    T_target_inv = torch.linalg.inv(T_target_mat)

    # Homogeneous
    ones = torch.ones(points_3d.shape[0], 1, device=device)
    points_hom = torch.cat([points_3d, ones], dim=1)

    points_cam = (T_target_inv @ points_hom.T).T[:, :3]

    # Project to image
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    z_valid = Z > 0.1

    u = fx * X / (Z + 1e-8) + cx
    v = fy * Y / (Z + 1e-8) + cy

    # Check if in image bounds
    in_bounds = (u >= 0) & (u < W-1) & (v >= 0) & (v < H-1) & z_valid

    # Sample target depth
    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

    depth_sampled = F.grid_sample(
        depth_target.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze()  # [N]

    # Compute depth error
    depth_error = torch.abs(Z - depth_sampled)
    relative_error = depth_error / (Z + 1e-8)

    # Inliers: relative depth error < 5%
    inlier_mask = in_bounds & (relative_error < 0.05)

    if inlier_mask.sum() > 0:
        error = relative_error[inlier_mask].mean().item()
    else:
        error = float('inf')

    return error, inlier_mask


class ScaleShiftEstimator:
    """
    Estimate scale and shift parameters for DA3 depth alignment.

    Uses multi-view triangulation and geometric validation.
    """

    def __init__(self, min_triangulation_pairs=3, min_parallax=0.02,
                 inlier_threshold=0.05, temporal_alpha=0.7):
        """
        Args:
            min_triangulation_pairs: Minimum number of frame pairs for robust estimation
            min_parallax: Minimum parallax angle in radians
            inlier_threshold: Inlier threshold for reprojection error
            temporal_alpha: Temporal smoothing factor (higher = more smoothing)
        """
        self.min_triangulation_pairs = min_triangulation_pairs
        self.min_parallax = min_parallax
        self.inlier_threshold = inlier_threshold
        self.temporal_alpha = temporal_alpha

        self.scale_history = []
        self.shift_history = []

    def estimate_scale_shift(self, da3_depth, droid_depth, da3_pose, droid_pose,
                            neighbor_frames, intrinsics):
        """
        Estimate scale and shift for a single keyframe.

        Args:
            da3_depth: DA3 depth map [H, W]
            droid_depth: DROID depth map [H, W]
            da3_pose: DA3 pose [7] (may be None if DA3 doesn't provide pose)
            droid_pose: DROID pose [7]
            neighbor_frames: List of (da3_depth_j, droid_depth_j, droid_pose_j, intrinsics_j)
            intrinsics: Camera intrinsics [4]

        Returns:
            scale: Estimated scale factor
            shift: Estimated shift factor
            confidence: Confidence score [0, 1]
        """

        if len(neighbor_frames) < self.min_triangulation_pairs:
            return self._get_smoothed_params(1.0, 0.0, 0.0)

        # Collect scale-shift estimates from multiple frame pairs
        scales = []
        shifts = []
        confidences = []

        for (da3_depth_j, droid_depth_j, droid_pose_j, intrinsics_j) in neighbor_frames:
            try:
                # Triangulate points using DA3 depth in frame i
                points_da3, valid_da3 = triangulate_points(
                    da3_depth, droid_pose, droid_pose_j, intrinsics,
                    self.min_parallax
                )

                # Triangulate points using DROID depth in frame i
                points_droid, valid_droid = triangulate_points(
                    droid_depth, droid_pose, droid_pose_j, intrinsics,
                    self.min_parallax
                )

                # Use common valid points
                valid = valid_da3 & valid_droid
                if valid.sum() < 100:
                    continue

                points_da3_valid = points_da3[valid]
                points_droid_valid = points_droid[valid]

                # Compute scale via Umeyama alignment
                scale, R, t = umeyama_alignment(points_da3_valid, points_droid_valid)

                # Validate via reprojection
                points_aligned = scale * (R @ points_da3_valid.T).T + t
                error, inlier_mask = compute_reprojection_error(
                    points_aligned, droid_depth_j, droid_pose_j, intrinsics_j
                )

                inlier_ratio = inlier_mask.sum().item() / len(inlier_mask)

                if inlier_ratio > 0.5 and error < self.inlier_threshold:
                    # Estimate shift via least squares on aligned points
                    # d_metric = s * d_da3 + t
                    # Using: d_droid_valid = scale * d_da3_valid + shift
                    da3_depths_valid = da3_depth[valid]
                    droid_depths_valid = droid_depth[valid]

                    # Least squares: minimize ||d_droid - (s * d_da3 + t)||^2
                    # This is a simple linear regression problem
                    A = torch.stack([da3_depths_valid, torch.ones_like(da3_depths_valid)], dim=1)
                    b = droid_depths_valid

                    # Solve normal equations: (A^T A) x = A^T b
                    AtA = A.T @ A
                    Atb = A.T @ b
                    x = torch.linalg.solve(AtA, Atb)

                    scale_ls, shift_ls = x[0].item(), x[1].item()

                    # Use least-squares estimate (more accurate for affine model)
                    scales.append(scale_ls)
                    shifts.append(shift_ls)
                    confidences.append(inlier_ratio)

            except Exception as e:
                print(f"[ScaleShiftEstimator] Warning: Failed on frame pair: {e}")
                continue

        # Robust fusion
        if len(scales) == 0:
            return self._get_smoothed_params(1.0, 0.0, 0.0)

        scales = torch.tensor(scales, device=da3_depth.device)
        shifts = torch.tensor(shifts, device=da3_depth.device)
        confidences = torch.tensor(confidences, device=da3_depth.device)

        # Weighted median
        scale = self._weighted_median(scales, confidences)
        shift = self._weighted_median(shifts, confidences)
        confidence = confidences.mean().item()

        return self._get_smoothed_params(scale.item(), shift.item(), confidence)

    def _weighted_median(self, values, weights):
        """Compute weighted median."""
        sorted_idx = torch.argsort(values)
        sorted_vals = values[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cumsum = torch.cumsum(sorted_weights, dim=0)
        total = cumsum[-1]

        median_idx = (cumsum >= total / 2.0).nonzero(as_tuple=True)[0][0]
        return sorted_vals[median_idx]

    def _get_smoothed_params(self, scale, shift, confidence):
        """Apply temporal smoothing."""
        self.scale_history.append(scale)
        self.shift_history.append(shift)

        # Keep recent history
        if len(self.scale_history) > 10:
            self.scale_history.pop(0)
            self.shift_history.pop(0)

        # Exponential moving average
        if len(self.scale_history) == 1:
            return scale, shift, confidence

        smoothed_scale = self.temporal_alpha * scale + (1 - self.temporal_alpha) * self.scale_history[-2]
        smoothed_shift = self.temporal_alpha * shift + (1 - self.temporal_alpha) * self.shift_history[-2]

        return smoothed_scale, smoothed_shift, confidence
