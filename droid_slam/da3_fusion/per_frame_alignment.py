"""
Per-Frame Scale-Shift Alignment for DA3 Depth Maps (Professor-Level)

Addresses the fundamental problem: DA3 depth maps have different affine parameters
per frame, but we were using a global scale. This causes misalignment.

Solution: Estimate per-frame (scale, shift) using DROID's triangulated sparse depth.

References:
- Metric3D (ICCV 2023): Per-image affine-invariant to metric conversion
- NICER-SLAM (CVPR 2024): Monocular depth alignment via triangulation
- MonoGS (CVPR 2024): Per-frame affine parameters in 3DGS optimization

Author: Professor-level AI Research
Date: 2025-12-05
"""

import torch
import torch.nn.functional as F
import numpy as np
from lietorch import SE3


class PerFrameDepthAligner:
    """
    Aligns each DA3 depth map to DROID's metric space using per-frame scale/shift.

    Key Innovation: Instead of global scale, we estimate (scale_i, shift_i) for
    each frame i by comparing DA3 depth with DROID's triangulated sparse depth.
    """

    def __init__(self, min_sparse_points=50, ransac_iterations=100, inlier_threshold=0.1):
        """
        Args:
            min_sparse_points: Minimum triangulated points needed for alignment
            ransac_iterations: RANSAC iterations for robust estimation
            inlier_threshold: Threshold for RANSAC inliers (relative error)
        """
        self.min_sparse_points = min_sparse_points
        self.ransac_iterations = ransac_iterations
        self.inlier_threshold = inlier_threshold

    def align_frame(self, frame_idx, da3_depth, droid_depth, droid_weight, intrinsics, poses, target_shape=None):
        """
        Align a single DA3 depth map to DROID's metric space.

        Strategy:
        1. Use DROID's inverse depth as "sparse metric depth" (it's already optimized)
        2. Sample at high-confidence pixels (where droid_weight is high)
        3. Robustly estimate (scale, shift) via RANSAC
        4. Apply to full DA3 depth map

        Args:
            frame_idx: Frame index
            da3_depth: DA3 depth map [H, W] (affine-invariant)
            droid_depth: DROID inverse depth [H, W] (metric, optimized)
            droid_weight: DROID confidence [H, W]
            intrinsics: Camera intrinsics [fx, fy, cx, cy]
            poses: All poses [N, 7] (SE3)
            target_shape: Target output shape (H_target, W_target), if None use da3_depth.shape

        Returns:
            aligned_depth: Metric-aligned depth [H_target, W_target]
            scale: Estimated scale
            shift: Estimated shift
            confidence: Alignment confidence [0, 1]
        """
        H_orig, W_orig = da3_depth.shape if target_shape is None else target_shape
        device = da3_depth.device

        # Step 0: Ensure DA3 depth matches DROID resolution
        H_droid, W_droid = droid_depth.shape
        if da3_depth.shape != (H_droid, W_droid):
            # Resize DA3 depth to match DROID resolution
            da3_depth = F.interpolate(
                da3_depth.unsqueeze(0).unsqueeze(0),
                size=(H_droid, W_droid),
                mode='bilinear',
                align_corners=False
            )[0, 0]

        # Step 1: Get sparse "ground truth" depth from DROID
        # DROID stores inverse depth, so convert: d = 1 / d_inv
        droid_depth_metric = 1.0 / (droid_depth + 1e-8)  # [H, W]

        # Step 2: Select high-confidence pixels for alignment
        # Use top 20% confidence pixels to avoid outliers
        weight_threshold = torch.quantile(droid_weight, 0.8)
        valid_mask = (droid_weight > weight_threshold) & (da3_depth > 0) & (droid_depth_metric > 0.1) & (droid_depth_metric < 100)

        if valid_mask.sum() < self.min_sparse_points:
            # Not enough sparse points, return unaligned with low confidence
            return da3_depth, 1.0, 0.0, 0.0

        # Step 3: Sample depths at valid pixels
        da3_sparse = da3_depth[valid_mask]  # [N_sparse]
        droid_sparse = droid_depth_metric[valid_mask]  # [N_sparse]

        # Step 4: Robust scale-shift estimation with RANSAC
        scale, shift, inlier_ratio = self._estimate_scale_shift_ransac(
            da3_sparse.cpu().numpy(),
            droid_sparse.cpu().numpy()
        )

        # Step 5: Apply alignment to full depth map
        aligned_depth = scale * da3_depth + shift

        # Step 6: Sanity check - clip to reasonable range
        aligned_depth = torch.clamp(aligned_depth, min=0.1, max=100.0)

        # Step 7: Resize back to original/target resolution if needed
        if aligned_depth.shape != (H_orig, W_orig):
            aligned_depth = F.interpolate(
                aligned_depth.unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )[0, 0]

        return aligned_depth, scale, shift, inlier_ratio

    def _estimate_scale_shift_ransac(self, da3_sparse, droid_sparse):
        """
        Robust scale-shift estimation using RANSAC.

        Model: d_metric = s * d_da3 + t

        Args:
            da3_sparse: DA3 depths [N]
            droid_sparse: DROID metric depths [N]

        Returns:
            scale: Best scale
            shift: Best shift
            inlier_ratio: Fraction of inliers
        """
        N = len(da3_sparse)
        if N < 10:
            # Fallback to least squares if too few points
            return self._least_squares_fit(da3_sparse, droid_sparse), 0.0

        best_scale = 1.0
        best_shift = 0.0
        best_inliers = 0

        # RANSAC loop
        for _ in range(self.ransac_iterations):
            # Sample 2 random points (minimum for affine model)
            idx = np.random.choice(N, size=2, replace=False)
            x1, x2 = da3_sparse[idx]
            y1, y2 = droid_sparse[idx]

            # Solve for scale and shift
            # y1 = s*x1 + t
            # y2 = s*x2 + t
            # => s = (y2 - y1) / (x2 - x1)
            # => t = y1 - s*x1
            if abs(x2 - x1) < 1e-6:
                continue

            s = (y2 - y1) / (x2 - x1)
            t = y1 - s * x1

            # Sanity check
            if s <= 0 or s > 10:  # Scale should be positive and reasonable
                continue

            # Count inliers
            predicted = s * da3_sparse + t
            errors = np.abs(predicted - droid_sparse) / (droid_sparse + 1e-6)
            inliers = (errors < self.inlier_threshold).sum()

            if inliers > best_inliers:
                best_inliers = inliers
                best_scale = s
                best_shift = t

        # Refine with least squares on inliers
        if best_inliers > 10:
            predicted = best_scale * da3_sparse + best_shift
            errors = np.abs(predicted - droid_sparse) / (droid_sparse + 1e-6)
            inlier_mask = errors < self.inlier_threshold

            if inlier_mask.sum() > 2:
                best_scale, best_shift = self._least_squares_fit(
                    da3_sparse[inlier_mask],
                    droid_sparse[inlier_mask]
                )

        inlier_ratio = best_inliers / N
        return best_scale, best_shift, inlier_ratio

    def _least_squares_fit(self, x, y):
        """
        Least squares fit: y = s*x + t

        Returns:
            scale, shift
        """
        N = len(x)
        if N < 2:
            return 1.0, 0.0

        # Build linear system: [x, 1] @ [s, t]^T = y
        A = np.stack([x, np.ones(N)], axis=1)  # [N, 2]
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        scale, shift = params[0], params[1]

        # Sanity check
        if scale <= 0 or scale > 10:
            scale = np.median(y / (x + 1e-6))  # Fallback to median ratio
            shift = 0.0

        return scale, shift

    def align_with_multiview_consistency(self, frame_idx, da3_depth, aligned_depth,
                                        poses, intrinsics, neighbor_indices, all_da3_depths,
                                        all_aligned_depths, consistency_threshold=0.05):
        """
        Refine alignment using multi-view geometric consistency.

        For each neighbor frame:
        1. Warp neighbor's aligned depth to current frame
        2. Check consistency with current aligned depth
        3. Filter out inconsistent pixels

        Args:
            frame_idx: Current frame index
            da3_depth: Original DA3 depth [H, W]
            aligned_depth: Initially aligned depth [H, W]
            poses: All poses [N, 7]
            intrinsics: [fx, fy, cx, cy]
            neighbor_indices: Indices of neighbor frames
            all_da3_depths: All DA3 depths dict
            all_aligned_depths: All aligned depths dict
            consistency_threshold: Relative error threshold

        Returns:
            refined_depth: Refined depth with consistency filter [H, W]
            valid_mask: Binary mask of consistent pixels [H, W]
        """
        H, W = aligned_depth.shape
        device = aligned_depth.device

        # Initialize consistency voting
        consistency_count = torch.zeros_like(aligned_depth)  # How many neighbors agree
        depth_sum = torch.zeros_like(aligned_depth)

        # Get current pose
        T_world_curr = SE3(poses[frame_idx:frame_idx+1]).matrix()[0]  # [4, 4]

        for neighbor_idx in neighbor_indices:
            if neighbor_idx == frame_idx:
                continue
            if neighbor_idx not in all_aligned_depths:
                continue

            # Get neighbor's aligned depth
            depth_neighbor = all_aligned_depths[neighbor_idx]  # [H, W]
            if depth_neighbor is None:
                continue

            # Get neighbor pose
            T_world_neighbor = SE3(poses[neighbor_idx:neighbor_idx+1]).matrix()[0]  # [4, 4]

            # Warp neighbor depth to current frame
            depth_warped = self._warp_depth(
                depth_neighbor,
                T_src=T_world_neighbor,
                T_tgt=T_world_curr,
                intrinsics=intrinsics
            )

            # Check consistency
            valid_warped = depth_warped > 0
            error = torch.abs(depth_warped - aligned_depth) / (aligned_depth + 1e-6)
            consistent = (error < consistency_threshold) & valid_warped

            # Vote
            consistency_count[consistent] += 1
            depth_sum[consistent] += depth_warped[consistent]

        # Require at least 2 neighbors to agree
        valid_mask = consistency_count >= 2

        # Fuse depths where consistent
        refined_depth = aligned_depth.clone()
        if valid_mask.sum() > 0:
            # Average with neighbors
            fused = depth_sum[valid_mask] / consistency_count[valid_mask]
            refined_depth[valid_mask] = 0.7 * aligned_depth[valid_mask] + 0.3 * fused

        # Mark inconsistent pixels as invalid
        refined_depth[~valid_mask] = 0.0

        return refined_depth, valid_mask

    def _warp_depth(self, depth_src, T_src, T_tgt, intrinsics):
        """
        Warp depth map from source to target frame.

        Args:
            depth_src: Source depth [H, W]
            T_src: Source pose (world to camera) [4, 4]
            T_tgt: Target pose (world to camera) [4, 4]
            intrinsics: [fx, fy, cx, cy]

        Returns:
            depth_warped: Warped depth [H, W]
        """
        H, W = depth_src.shape
        device = depth_src.device

        # Create pixel grid
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        # Unproject source pixels to 3D
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        X = (x - cx) * depth_src / fx
        Y = (y - cy) * depth_src / fy
        Z = depth_src

        # Stack to [H, W, 3]
        points_src_cam = torch.stack([X, Y, Z], dim=-1)  # [H, W, 3]

        # Transform to world
        T_src_inv = torch.inverse(T_src)  # Camera to world
        points_src_hom = torch.cat([
            points_src_cam,
            torch.ones(H, W, 1, device=device)
        ], dim=-1)  # [H, W, 4]

        points_world = (T_src_inv @ points_src_hom.reshape(-1, 4).T).T.reshape(H, W, 4)

        # Transform to target camera
        points_tgt_hom = (T_tgt @ points_world.reshape(-1, 4).T).T.reshape(H, W, 4)
        points_tgt_cam = points_tgt_hom[:, :, :3]  # [H, W, 3]

        # Project to target image
        Z_tgt = points_tgt_cam[:, :, 2]
        x_tgt = fx * points_tgt_cam[:, :, 0] / (Z_tgt + 1e-8) + cx
        y_tgt = fy * points_tgt_cam[:, :, 1] / (Z_tgt + 1e-8) + cy

        # Sample depth at target coordinates using grid_sample
        # Normalize to [-1, 1]
        x_norm = 2 * x_tgt / (W - 1) - 1
        y_norm = 2 * y_tgt / (H - 1) - 1
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        # Sample
        depth_warped = F.grid_sample(
            Z_tgt.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )[0, 0]  # [H, W]

        # Mark invalid (outside bounds or behind camera)
        valid = (x_tgt >= 0) & (x_tgt < W) & (y_tgt >= 0) & (y_tgt < H) & (Z_tgt > 0.1)
        depth_warped[~valid] = 0.0

        return depth_warped
