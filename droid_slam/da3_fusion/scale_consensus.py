"""
Multi-view scale consensus for aligning affine-invariant depth to metric scale
"""
import torch
import torch.nn.functional as F
from lietorch import SE3


class MultiViewScaleConsensus:
    """
    Estimates metric scale by comparing DA3 poses with DROID poses.

    Key idea:
    - DA3 predicts relative poses (scale-ambiguous)
    - DROID estimates metric poses
    - Compare relative motions to infer scale

    Uses RANSAC-like robust estimation with rotation consistency weighting.
    """

    def __init__(self, history_size=10, min_motion_thresh=0.01):
        """
        Args:
            history_size: Number of scale estimates to keep for smoothing
            min_motion_thresh: Minimum translation magnitude to consider
        """
        self.scale_history = []
        self.history_size = history_size
        self.min_motion_thresh = min_motion_thresh

    def estimate_scale(self, da3_poses, droid_poses, frame_indices=None):
        """
        Estimate scale factor from pose correspondences.

        Args:
            da3_poses: List of SE3 poses from DA3 [N, 7] or list of [7]
            droid_poses: List of SE3 poses from DROID [N, 7] or list of [7]
            frame_indices: Optional frame indices for debugging

        Returns:
            float: Estimated scale factor
        """
        # Convert to tensors if needed
        if isinstance(da3_poses, list):
            # Check if first element is SE3 or tensor
            if hasattr(da3_poses[0], 'data'):
                da3_poses = torch.stack([p.data for p in da3_poses])
            else:
                da3_poses = torch.stack(da3_poses)

        if isinstance(droid_poses, list):
            if hasattr(droid_poses[0], 'data'):
                droid_poses = torch.stack([p.data for p in droid_poses])
            else:
                droid_poses = torch.stack(droid_poses)

        # Handle None poses (DA3 might not predict poses)
        if da3_poses is None or len(da3_poses) < 2:
            return self._get_smoothed_scale(1.0)

        # Move to same device
        device = droid_poses.device
        da3_poses = da3_poses.to(device)

        N = len(da3_poses)
        if N < 2:
            return self._get_smoothed_scale(1.0)

        # Compute all pairwise relative motions
        scales = []
        confidences = []

        for i in range(N - 1):
            for j in range(i + 1, min(i + 4, N)):  # Look ahead up to 3 frames
                # DA3 relative pose
                T_da3_i = SE3(da3_poses[i:i+1])
                T_da3_j = SE3(da3_poses[j:j+1])
                T_da3_rel = T_da3_i.inv() * T_da3_j

                # DROID relative pose
                T_droid_i = SE3(droid_poses[i:i+1])
                T_droid_j = SE3(droid_poses[j:j+1])
                T_droid_rel = T_droid_i.inv() * T_droid_j

                # Extract translation from 4x4 matrix
                T_da3_mat = T_da3_rel.matrix()[0]  # [4, 4]
                T_droid_mat = T_droid_rel.matrix()[0]  # [4, 4]

                t_da3_vec = T_da3_mat[:3, 3]  # [3]
                t_droid_vec = T_droid_mat[:3, 3]  # [3]

                # Translation magnitudes
                t_da3 = t_da3_vec.norm()
                t_droid = t_droid_vec.norm()

                # Only use if sufficient motion
                if t_da3 > self.min_motion_thresh and t_droid > self.min_motion_thresh:
                    # Scale ratio
                    scale_ij = t_droid / (t_da3 + 1e-8)

                    # Confidence from rotation consistency
                    # If rotations agree, poses are more reliable
                    R_da3 = T_da3_mat[:3, :3]  # [3, 3]
                    R_droid = T_droid_mat[:3, :3]  # [3, 3]

                    # Compute rotation error using Frobenius norm
                    R_diff = R_droid @ R_da3.T  # Should be close to identity if consistent
                    rot_error = (R_diff - torch.eye(3, device=R_diff.device, dtype=R_diff.dtype)).norm()

                    # Exponential decay with rotation error
                    conf = torch.exp(-2.0 * rot_error)

                    scales.append(scale_ij.item())
                    confidences.append(conf.item())

        # Robust estimation
        if len(scales) == 0:
            return self._get_smoothed_scale(1.0)

        scales = torch.tensor(scales, device=device)
        confidences = torch.tensor(confidences, device=device)

        # Filter by confidence threshold
        high_conf_mask = confidences > 0.7
        if high_conf_mask.sum() > 0:
            filtered_scales = scales[high_conf_mask]
            filtered_confs = confidences[high_conf_mask]
        else:
            filtered_scales = scales
            filtered_confs = confidences

        # Weighted median for robustness
        scale = self._weighted_median(filtered_scales, filtered_confs)

        return self._get_smoothed_scale(scale.item())

    def _weighted_median(self, values, weights):
        """
        Compute weighted median (more robust than mean).

        Args:
            values: Tensor of values
            weights: Tensor of weights

        Returns:
            Weighted median value
        """
        # Sort by values
        sorted_idx = torch.argsort(values)
        sorted_vals = values[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Cumulative weights
        cumsum = torch.cumsum(sorted_weights, dim=0)
        total_weight = cumsum[-1]

        # Find median position
        median_pos = total_weight / 2.0
        median_idx = (cumsum >= median_pos).nonzero(as_tuple=True)[0][0]

        return sorted_vals[median_idx]

    def _get_smoothed_scale(self, new_scale):
        """
        Apply temporal smoothing to scale estimates.

        Args:
            new_scale: New scale estimate

        Returns:
            Smoothed scale
        """
        # Add to history
        self.scale_history.append(new_scale)

        # Keep only recent history
        if len(self.scale_history) > self.history_size:
            self.scale_history.pop(0)

        # Median filter for robustness
        return float(torch.median(torch.tensor(self.scale_history)).item())

    def reset(self):
        """Reset scale history."""
        self.scale_history = []


def align_depth_to_metric_scale(da3_depth, scale):
    """
    Convert affine-invariant depth to metric depth using estimated scale.

    DA3 depth has form: d_da3 = a * d_true + b
    We estimate scale 's' such that: d_metric = d_da3 / s

    Args:
        da3_depth: Depth map from DA3 [H, W]
        scale: Estimated scale factor

    Returns:
        Metric depth map [H, W]
    """
    return da3_depth / (scale + 1e-8)


def compute_disparity_from_depth(depth, min_disp=0.001):
    """
    Convert depth to disparity.

    Args:
        depth: Depth map [H, W]
        min_disp: Minimum disparity to avoid division by zero

    Returns:
        Disparity map [H, W]
    """
    disp = 1.0 / (depth + 1e-8)
    disp = torch.clamp(disp, min=min_disp)
    return disp
