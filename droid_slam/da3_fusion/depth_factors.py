"""
Depth Factor Integration for Bundle Adjustment

This module implements depth observation factors that directly constrain
the bundle adjustment optimization using DA3 depth priors.

Key Innovation:
    Instead of passively injecting depth into disps_sens, we add explicit
    depth factors to the factor graph with confidence-weighted residuals.

Mathematical Formulation:
    Total Energy: E = E_photometric + λ_depth * E_depth

    E_depth = Σ_i Σ_p w(p) * ρ(d_droid(p) - d_prior(p))

    where:
    - w(p) = confidence from DA3
    - ρ() = Huber loss (robust to outliers)
    - d_prior(p) = s * d_da3(p) + t (scaled DA3 depth)
    - d_droid(p) = 1 / disp_droid(p) (current DROID depth)

Jacobians:
    We need ∂E_depth/∂poses and ∂E_depth/∂disps for Gauss-Newton optimization.

    ∂d_droid/∂disp = -1 / disp^2
    ∂d_droid/∂pose = 0 (depth is local to camera frame)

    Therefore:
    ∂E_depth/∂disp = w * ρ'(...) * (-1 / disp^2)
    ∂E_depth/∂pose = 0 (depth factors don't constrain pose directly,
                         but they influence pose through coupling in full Hessian)
"""

import torch
import torch.nn.functional as F


def huber_loss(x, delta=0.1):
    """
    Huber loss: robust to outliers.

    Args:
        x: Residual tensor
        delta: Threshold for quadratic vs linear

    Returns:
        loss: Huber loss
        weight: Derivative weight (for linearization)
    """
    abs_x = torch.abs(x)
    quadratic = abs_x <= delta
    linear = ~quadratic

    loss = torch.where(
        quadratic,
        0.5 * x ** 2,
        delta * (abs_x - 0.5 * delta)
    )

    # Derivative: ρ'(x) = x if |x| <= delta else delta * sign(x)
    weight = torch.where(
        quadratic,
        torch.ones_like(x),
        delta / (abs_x + 1e-8)
    )

    return loss, weight


class DepthFactorManager:
    """
    Manages depth factors for integration into DROID's factor graph.

    This class maintains depth priors from DA3 and computes residuals + Jacobians
    for bundle adjustment.
    """

    def __init__(self, video, lambda_depth=0.1, huber_delta=0.1):
        """
        Args:
            video: DepthVideo object (shared state)
            lambda_depth: Weight for depth factors relative to photometric
            huber_delta: Threshold for Huber loss
        """
        self.video = video
        self.lambda_depth = lambda_depth
        self.huber_delta = huber_delta

        # Store depth priors: {frame_idx: (depth_prior, confidence, scale, shift)}
        self.depth_priors = {}

    def add_depth_prior(self, frame_idx, da3_depth, da3_conf, scale, shift):
        """
        Add DA3 depth prior for a frame.

        Args:
            frame_idx: Frame index
            da3_depth: DA3 depth map [H, W] at 1/8 resolution
            da3_conf: DA3 confidence map [H, W]
            scale: Scale factor
            shift: Shift factor
        """
        device = self.video.disps.device

        # Ensure correct resolution (1/8 of full image)
        H, W = self.video.ht // 8, self.video.wd // 8

        if da3_depth.shape != (H, W):
            da3_depth = F.interpolate(
                da3_depth.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze()

        if da3_conf.shape != (H, W):
            da3_conf = F.interpolate(
                da3_conf.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze()

        # Compute metric depth prior
        depth_prior = scale * da3_depth + shift

        # Clamp to reasonable range
        depth_prior = torch.clamp(depth_prior, min=0.2, max=100.0)

        self.depth_priors[frame_idx] = {
            'depth': depth_prior.to(device),
            'confidence': da3_conf.to(device),
            'scale': scale,
            'shift': shift
        }

    def compute_depth_residuals(self, frame_indices=None):
        """
        Compute depth residuals for frames with priors.

        Args:
            frame_indices: List of frame indices to compute (default: all with priors)

        Returns:
            residuals: Dict mapping frame_idx -> residual tensor [H, W]
            weights: Dict mapping frame_idx -> weight tensor [H, W]
            loss: Total depth loss (scalar)
        """
        if frame_indices is None:
            frame_indices = list(self.depth_priors.keys())

        residuals = {}
        weights = {}
        total_loss = 0.0
        total_weight = 0.0

        for idx in frame_indices:
            if idx not in self.depth_priors:
                continue

            prior = self.depth_priors[idx]
            depth_prior = prior['depth']
            confidence = prior['confidence']

            # Current DROID disparity
            disp_droid = self.video.disps[idx]  # [H, W]

            # Convert to depth
            depth_droid = 1.0 / (disp_droid + 1e-6)

            # Residual
            residual = depth_droid - depth_prior  # [H, W]

            # Confidence-weighted Huber loss
            loss_per_pixel, huber_weight = huber_loss(residual, delta=self.huber_delta)

            # Combine with DA3 confidence
            weight = confidence * huber_weight

            # Mask out low-confidence regions
            valid_mask = confidence > 0.3
            weight = weight * valid_mask.float()

            # Accumulate loss
            weighted_loss = (loss_per_pixel * weight).sum()
            total_loss += weighted_loss
            total_weight += weight.sum()

            residuals[idx] = residual
            weights[idx] = weight

        # Normalize loss
        if total_weight > 0:
            total_loss = self.lambda_depth * total_loss / (total_weight + 1e-8)
        else:
            total_loss = torch.tensor(0.0, device=self.video.disps.device)

        return residuals, weights, total_loss

    def compute_depth_jacobians(self, frame_indices=None):
        """
        Compute Jacobians of depth residuals w.r.t. disparity.

        Since d_droid = 1 / disp, we have:
        ∂d_droid/∂disp = -1 / disp^2

        Args:
            frame_indices: Frame indices to compute

        Returns:
            jacobians: Dict mapping frame_idx -> Jacobian tensor [H, W]
        """
        if frame_indices is None:
            frame_indices = list(self.depth_priors.keys())

        jacobians = {}

        for idx in frame_indices:
            if idx not in self.depth_priors:
                continue

            disp_droid = self.video.disps[idx]

            # ∂d/∂disp = -1 / disp^2
            jac = -1.0 / (disp_droid ** 2 + 1e-6)

            jacobians[idx] = jac

        return jacobians

    def apply_depth_update(self, frame_indices, delta_disps):
        """
        Apply depth factor updates to disparity.

        This is called during Gauss-Newton iterations.

        Args:
            frame_indices: Frame indices to update
            delta_disps: Dict mapping frame_idx -> disparity update [H, W]
        """
        for idx in frame_indices:
            if idx in delta_disps:
                # Apply update with damping
                self.video.disps[idx] += 0.1 * delta_disps[idx]

                # Clamp to valid range
                self.video.disps[idx] = torch.clamp(
                    self.video.disps[idx],
                    min=0.01,  # max depth = 100m
                    max=10.0   # min depth = 0.1m
                )

    def inject_depth_prior_to_video(self, frame_idx, conf_threshold=0.5):
        """
        Inject depth prior into video.disps_sens for DROID's internal use.

        This provides a good initialization before bundle adjustment.

        Args:
            frame_idx: Frame index
            conf_threshold: Minimum confidence to inject
        """
        if frame_idx not in self.depth_priors:
            return

        prior = self.depth_priors[frame_idx]
        depth_prior = prior['depth']
        confidence = prior['confidence']

        # Convert to disparity
        disp_prior = 1.0 / (depth_prior + 1e-6)

        # High-confidence mask
        high_conf = confidence > conf_threshold

        # Inject into disps_sens
        self.video.disps_sens[frame_idx][high_conf] = disp_prior[high_conf]

    def clear_old_priors(self, keep_recent=20):
        """
        Clear old depth priors to save memory.

        Args:
            keep_recent: Number of recent priors to keep
        """
        if len(self.depth_priors) <= keep_recent:
            return

        sorted_indices = sorted(self.depth_priors.keys())
        old_indices = sorted_indices[:-keep_recent]

        for idx in old_indices:
            del self.depth_priors[idx]

    def get_prior_status(self):
        """Get status string for debugging."""
        return f"DepthFactors: {len(self.depth_priors)} priors active"


def compute_hybrid_depth(droid_depth, da3_depth, da3_conf, scale, shift):
    """
    Compute hybrid depth map fusing DROID and DA3.

    Uses confidence-weighted blending with bilateral filtering at boundaries.

    Args:
        droid_depth: DROID depth [H, W]
        da3_depth: DA3 depth [H, W]
        da3_conf: DA3 confidence [H, W]
        scale: Scale factor
        shift: Shift factor

    Returns:
        hybrid_depth: Fused depth map [H, W]
    """
    device = droid_depth.device

    # Scale DA3 depth
    da3_scaled = scale * da3_depth + shift
    da3_scaled = torch.clamp(da3_scaled, min=0.2, max=100.0)

    # Confidence-weighted fusion
    # In high-confidence regions, trust DA3; in low-confidence, trust DROID
    weight_da3 = da3_conf
    weight_droid = 1.0 - da3_conf

    hybrid = weight_da3 * da3_scaled + weight_droid * droid_depth

    # Apply bilateral filtering to smooth boundaries
    # (Simple box filter for now; can be replaced with proper bilateral)
    kernel_size = 5
    padding = kernel_size // 2

    hybrid_smooth = F.avg_pool2d(
        hybrid.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    ).squeeze()

    # Use smooth version at boundary regions (medium confidence)
    boundary_mask = (da3_conf > 0.3) & (da3_conf < 0.7)
    hybrid[boundary_mask] = hybrid_smooth[boundary_mask]

    return hybrid
