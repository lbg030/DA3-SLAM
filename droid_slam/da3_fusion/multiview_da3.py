"""
Multi-View DA3 Inference with DROID Geometric Context (Professor-Level Innovation)

Key Innovation: Instead of running DA3 on single frames, we feed multiple frames
with DROID's accurate poses to leverage DA3's native multi-view capability.

This addresses the fundamental scale ambiguity problem by letting DA3 "see" camera motion.

References:
- depth_anything_3/model/da3.py:100-124 (multi-view forward pass)
- DROID-SLAM factor graph (temporal consistency)

Author: Professor-level AI Research
Date: 2025-12-05
"""

import torch
import torch.nn.functional as F
from lietorch import SE3


class MultiViewDA3Inference:
    """
    Runs DA3 depth estimation with multi-view geometric context from DROID.

    Unlike single-view DA3 (scale-ambiguous), this provides:
    1. Camera motion context → better scale estimation
    2. Multi-view consistency → reduced artifacts
    3. Temporal coherence → stable depth sequences
    4. Occlusion reasoning → handles dynamic objects
    """

    def __init__(self, da3_model, window_size=3):
        """
        Args:
            da3_model: Depth Anything V3 model
            window_size: Number of frames in temporal window (must be odd)
        """
        self.da3_model = da3_model
        self.window_size = window_size
        assert window_size % 2 == 1, "Window size must be odd (for center frame)"

        print(f"[MultiViewDA3] Initialized with window_size={window_size}")

    def infer(self, frame_idx, video, return_features=False):
        """
        Run DA3 with multi-view context from DROID.

        Args:
            frame_idx: Target frame index
            video: DepthVideo instance with DROID poses
            return_features: Whether to return auxiliary features

        Returns:
            dict with keys: 'depth', 'confidence', 'features' (optional)
        """
        # Step 1: Collect temporal window
        indices = self._get_temporal_window(frame_idx, video)
        if len(indices) < 2:
            print(f"[MultiViewDA3] Warning: Not enough neighbors for frame {frame_idx}, using single-view")
            return self._single_view_fallback(frame_idx, video, return_features)

        # Step 2: Prepare multi-view input
        images, extrinsics, intrinsics = self._prepare_multiview_input(indices, video)

        # Get original image resolution for resizing depth back
        H_orig, W_orig = video.images[frame_idx].shape[-2:]

        # Step 3: Run DA3 with geometric context
        with torch.no_grad():
            output = self.da3_model(
                images,  # [1, N, 3, H, W]
                extrinsics=extrinsics,  # [1, N, 4, 4]
                intrinsics=intrinsics,  # [1, N, 3, 3]
                export_feat_layers=[11, 17, 23] if return_features else []
            )

        # Step 4: Extract results for target frame
        center_idx = indices.index(frame_idx)
        depth = output['depth'][0, center_idx]  # [H_target, W_target]
        confidence = output.get('depth_conf', torch.ones_like(depth))[0, center_idx]

        # Resize depth and confidence back to original resolution
        H_target, W_target = depth.shape
        if H_orig != H_target or W_orig != W_target:
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )[0, 0]

            confidence = F.interpolate(
                confidence.unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )[0, 0]

        result = {
            'depth': depth,
            'confidence': confidence
        }

        if return_features:
            features = {}
            if 'aux' in output and output['aux'] is not None:
                for key, val in output['aux'].items():
                    if key.startswith('feat_layer_'):
                        features[key] = val[0, center_idx]
            result['features'] = features

        return result

    def _get_temporal_window(self, frame_idx, video):
        """
        Get indices of frames in temporal window around target frame.

        Strategy:
        1. Prefer symmetric window: [frame_idx-1, frame_idx, frame_idx+1]
        2. Skip frames without valid poses
        3. Ensure minimum baseline between frames

        Returns:
            List of frame indices (sorted)
        """
        half_window = self.window_size // 2
        candidates = []

        # Collect candidates
        for offset in range(-half_window, half_window + 1):
            idx = frame_idx + offset
            if 0 <= idx < video.counter.value:
                # Check if pose is valid (non-zero)
                if video.poses[idx].abs().sum() > 0:
                    candidates.append(idx)

        # Filter by baseline (ensure sufficient motion)
        if len(candidates) > 1:
            # CRITICAL: Always ensure target frame_idx is included!
            if frame_idx not in candidates:
                return candidates  # Fallback: return all candidates if target missing

            # Start with target frame
            filtered = [frame_idx]
            poses_se3 = SE3(video.poses[torch.tensor(candidates)])
            T_world_cam = poses_se3.matrix()  # [N, 4, 4]

            # Get target frame's position
            target_idx_in_candidates = candidates.index(frame_idx)
            t_target = T_world_cam[target_idx_in_candidates, :3, 3]

            # Add neighbors with sufficient baseline
            for i, cand_idx in enumerate(candidates):
                if cand_idx == frame_idx:
                    continue  # Already added

                t_i = T_world_cam[i, :3, 3]
                baseline = torch.norm(t_i - t_target).item()

                if baseline > 0.02:  # Minimum 2cm baseline from target
                    filtered.append(cand_idx)

                if len(filtered) >= self.window_size:
                    break

            # Sort to maintain temporal order
            filtered.sort()
            return filtered

        return candidates

    def _prepare_multiview_input(self, indices, video):
        """
        Prepare multi-view input tensors for DA3.

        Returns:
            images: [1, N, 3, H, W]
            extrinsics: [1, N, 4, 4] (relative to reference frame)
            intrinsics: [1, N, 3, 3] (scaled to full resolution)
        """
        N = len(indices)
        device = video.images.device

        # Get images (already at full resolution)
        images = video.images[indices].float()  # [N, 3, H, W], convert to float
        H_orig, W_orig = images.shape[-2:]

        # DA3 requires input dimensions to be multiples of 14 (ViT patch size)
        H_target = ((H_orig + 13) // 14) * 14
        W_target = ((W_orig + 13) // 14) * 14

        # Resize if needed
        if H_orig != H_target or W_orig != W_target:
            images = F.interpolate(
                images,  # [N, 3, H, W]
                size=(H_target, W_target),
                mode='bilinear',
                align_corners=False
            )

        images = images.unsqueeze(0) / 255.0  # [1, N, 3, H_target, W_target], normalized

        # Get poses and convert to 4x4 matrices
        poses_se3 = video.poses[indices]  # [N, 7]
        T_world_cam = SE3(poses_se3).matrix()  # [N, 4, 4]

        # Compute relative extrinsics (frame 0 as reference)
        T_ref_world = torch.inverse(T_world_cam[0:1])  # [1, 4, 4]
        extrinsics_rel = T_ref_world @ T_world_cam  # [N, 4, 4]
        extrinsics = extrinsics_rel.unsqueeze(0)  # [1, N, 4, 4]

        # Prepare intrinsics (scale from 1/8 to target resolution)
        K_droid = video.intrinsics[0]  # [4]: [fx, fy, cx, cy] at 1/8 resolution

        # CRITICAL: DROID intrinsics are at 1/8 resolution
        # Scale to target resolution
        scale_h = H_target / (H_orig / 8.0)  # Scale from 1/8 to target
        scale_w = W_target / (W_orig / 8.0)

        intrinsics_list = []
        for _ in range(N):
            K_3x3 = torch.tensor([
                [K_droid[0] * scale_w, 0, K_droid[2] * scale_w],
                [0, K_droid[1] * scale_h, K_droid[3] * scale_h],
                [0, 0, 1]
            ], device=device, dtype=torch.float32)
            intrinsics_list.append(K_3x3)

        intrinsics = torch.stack(intrinsics_list, dim=0).unsqueeze(0)  # [1, N, 3, 3]

        return images, extrinsics, intrinsics

    def _single_view_fallback(self, frame_idx, video, return_features=False):
        """
        Fallback to single-view DA3 if multi-view not available.

        Returns:
            dict with keys: 'depth', 'confidence', 'features' (optional)
        """
        image = video.images[frame_idx].float()  # [3, H, W], convert to float
        H_orig, W_orig = image.shape[-2:]

        # DA3 requires input dimensions to be multiples of 14 (ViT patch size)
        H_target = ((H_orig + 13) // 14) * 14
        W_target = ((W_orig + 13) // 14) * 14

        # Resize if needed
        if H_orig != H_target or W_orig != W_target:
            image = F.interpolate(
                image.unsqueeze(0),  # [1, 3, H, W]
                size=(H_target, W_target),
                mode='bilinear',
                align_corners=False
            )[0]  # [3, H_target, W_target]

        image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H_target, W_target]

        with torch.no_grad():
            output = self.da3_model(
                image / 255.0,  # Normalize to [0, 1]
                export_feat_layers=[11, 17, 23] if return_features else []
            )

        depth = output['depth'][0, 0]  # [H_target, W_target]
        confidence = output.get('depth_conf', torch.ones_like(depth))[0, 0]

        # Resize back to original resolution
        if H_orig != H_target or W_orig != W_target:
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )[0, 0]

            confidence = F.interpolate(
                confidence.unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )[0, 0]

        result = {
            'depth': depth,
            'confidence': confidence
        }

        if return_features:
            features = {}
            if 'aux' in output and output['aux'] is not None:
                for key, val in output['aux'].items():
                    if key.startswith('feat_layer_'):
                        features[key] = val[0, 0]
            result['features'] = features

        return result

    def batch_infer(self, frame_indices, video):
        """
        Run multi-view DA3 on multiple frames efficiently.

        Args:
            frame_indices: List of frame indices to process
            video: DepthVideo instance

        Returns:
            depths: Dict {frame_idx: depth [H, W]}
            confidences: Dict {frame_idx: confidence [H, W]}
        """
        depths = {}
        confidences = {}

        for idx in frame_indices:
            depth, conf = self.infer(idx, video, return_features=False)
            depths[idx] = depth
            confidences[idx] = conf

        return depths, confidences


def compare_single_vs_multiview(frame_idx, video, da3_model, visualize=False):
    """
    Experimental function to compare single-view vs multi-view DA3.

    This is for research/debugging purposes.

    Args:
        frame_idx: Target frame
        video: DepthVideo instance
        da3_model: DA3 model
        visualize: Whether to save visualization

    Returns:
        Dict with comparison metrics
    """
    # Run single-view
    image = video.images[frame_idx].unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output_single = da3_model(image)
    depth_single = output_single['depth'][0, 0]

    # Run multi-view
    mv_engine = MultiViewDA3Inference(da3_model, window_size=3)
    depth_multi, conf_multi = mv_engine.infer(frame_idx, video)

    # Compute metrics
    metrics = {
        'depth_single_mean': depth_single.mean().item(),
        'depth_single_std': depth_single.std().item(),
        'depth_multi_mean': depth_multi.mean().item(),
        'depth_multi_std': depth_multi.std().item(),
        'conf_multi_mean': conf_multi.mean().item(),
        'relative_diff': (torch.abs(depth_multi - depth_single) / (depth_single + 1e-6)).mean().item()
    }

    if visualize:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(depth_single.cpu().numpy())
        axes[0].set_title(f'Single-view (mean={metrics["depth_single_mean"]:.2f})')

        axes[1].imshow(depth_multi.cpu().numpy())
        axes[1].set_title(f'Multi-view (mean={metrics["depth_multi_mean"]:.2f})')

        axes[2].imshow(conf_multi.cpu().numpy(), cmap='hot')
        axes[2].set_title(f'Confidence (mean={metrics["conf_multi_mean"]:.2f})')

        plt.tight_layout()
        plt.savefig(f'/tmp/da3_comparison_frame_{frame_idx}.png')
        plt.close()

        print(f"[Comparison] Saved visualization to /tmp/da3_comparison_frame_{frame_idx}.png")

    print(f"[Comparison] Frame {frame_idx}:")
    print(f"  Single-view: mean={metrics['depth_single_mean']:.3f}, std={metrics['depth_single_std']:.3f}")
    print(f"  Multi-view:  mean={metrics['depth_multi_mean']:.3f}, std={metrics['depth_multi_std']:.3f}")
    print(f"  Relative diff: {metrics['relative_diff']*100:.2f}%")

    return metrics
