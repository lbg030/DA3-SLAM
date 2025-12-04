"""
Pose refinement using DA3 depth as geometric constraint
"""
import torch
import torch.nn.functional as F
from lietorch import SE3
import geom.projective_ops as pops


class DepthGuidedPoseRefiner:
    """
    Refines DROID poses using DA3 depth as geometric constraints.

    Adds depth-based residuals to the photometric residuals
    in the factor graph optimization.
    """

    def __init__(self, video):
        """
        Args:
            video: DepthVideo instance
        """
        self.video = video

    def compute_geometric_residual(self, frame_i, frame_j, da3_depth_i, da3_conf_i, coords_grid):
        """
        Compute geometric residual between two frames using DA3 depth.

        Args:
            frame_i, frame_j: Frame indices
            da3_depth_i: DA3 depth for frame i [H, W]
            da3_conf_i: DA3 confidence for frame i [H, W]
            coords_grid: Coordinate grid [H, W, 2]

        Returns:
            geometric_target: Target coordinates from depth projection [H, W, 2]
            geometric_weight: Confidence weights [H, W, 2]
        """
        # Get current pose estimates
        pose_i = SE3(self.video.poses[frame_i:frame_i+1])
        pose_j = SE3(self.video.poses[frame_j:frame_j+1])
        intrinsics = self.video.intrinsics[0]

        # Resize DA3 depth to match coordinate grid
        H_grid, W_grid = coords_grid.shape[:2]
        H_depth, W_depth = da3_depth_i.shape

        if H_depth != H_grid or W_depth != W_grid:
            da3_depth_resized = F.interpolate(
                da3_depth_i[None, None],
                size=(H_grid, W_grid),
                mode='bilinear',
                align_corners=False
            )[0, 0]

            da3_conf_resized = F.interpolate(
                da3_conf_i[None, None],
                size=(H_grid, W_grid),
                mode='bilinear',
                align_corners=False
            )[0, 0]
        else:
            da3_depth_resized = da3_depth_i
            da3_conf_resized = da3_conf_i

        # Convert depth to disparity
        da3_disp = 1.0 / (da3_depth_resized + 1e-6)

        # Add batch and frame dimensions for pops
        da3_disp_batch = da3_disp[None, None]  # [1, 1, H, W]
        poses_batch = torch.stack([pose_i.data, pose_j.data])  # [2, 7]
        intrinsics_batch = intrinsics[None]  # [1, 4]

        # Create edge index
        ii = torch.tensor([0], device=da3_disp.device, dtype=torch.long)
        jj = torch.tensor([1], device=da3_disp.device, dtype=torch.long)

        # Use DROID's projective transform
        try:
            coords_proj, valid_mask = pops.projective_transform(
                SE3(poses_batch),
                da3_disp_batch,
                intrinsics_batch,
                ii,
                jj
            )
            # coords_proj: [1, 1, H, W, 2]
            # valid_mask: [1, 1, H, W, 1]

            geometric_target = coords_proj[0, 0]  # [H, W, 2]
            valid = valid_mask[0, 0, :, :, 0]  # [H, W]

        except Exception as e:
            print(f"Warning: projective_transform failed: {e}")
            # Fallback: use original coordinates
            geometric_target = coords_grid
            valid = torch.ones(H_grid, W_grid, device=coords_grid.device)

        # Weight based on DA3 confidence and validity
        conf_weight = da3_conf_resized * valid
        geometric_weight = torch.stack([conf_weight, conf_weight], dim=-1)  # [H, W, 2]

        return geometric_target, geometric_weight

    def add_depth_constraints_to_graph(self, graph, frame_idx, da3_data, scale):
        """
        Add DA3 depth constraints to factor graph.

        Args:
            graph: FactorGraph instance
            frame_idx: Frame index with DA3 data
            da3_data: Dict with 'depth' and 'conf'
            scale: Metric scale factor
        """
        # Get scaled metric depth
        depth_metric = da3_data['depth'] / scale
        conf = da3_data['conf']

        # Find related edges in graph
        related_i = (graph.ii == frame_idx).nonzero(as_tuple=True)[0]
        related_j = (graph.jj == frame_idx).nonzero(as_tuple=True)[0]

        coords_grid = graph.coords0  # [H, W, 2]

        # Process edges where frame_idx is source (ii)
        for edge_idx in related_i:
            i = graph.ii[edge_idx].item()
            j = graph.jj[edge_idx].item()

            if i == frame_idx:
                # Compute geometric residual
                geo_target, geo_weight = self.compute_geometric_residual(
                    i, j, depth_metric, conf, coords_grid
                )

                # Add to graph targets with reduced weight
                # (geometric is auxiliary to photometric)
                alpha = 0.05  # Weight for geometric vs photometric

                # Existing photometric target
                photo_target = graph.target[0, edge_idx]  # [H, W, 2]
                photo_weight = graph.weight[0, edge_idx]  # [H, W, 2]

                # Combined target (weighted average)
                combined_weight = photo_weight + alpha * geo_weight
                combined_target = (
                    photo_weight * photo_target + alpha * geo_weight * geo_target
                ) / (combined_weight + 1e-8)

                # Update graph
                graph.target[0, edge_idx] = combined_target
                graph.weight[0, edge_idx] = combined_weight


def inject_depth_prior_to_video(video, frame_idx, da3_depth, da3_conf, scale, conf_threshold=0.7):
    """
    Inject DA3 depth as prior into video.disps_sens.

    Args:
        video: DepthVideo instance
        frame_idx: Frame index
        da3_depth: DA3 depth map [H, W]
        da3_conf: DA3 confidence map [H, W]
        scale: Metric scale factor
        conf_threshold: Only use pixels above this confidence
    """
    # Convert to metric depth
    depth_metric = da3_depth / scale

    # Resize to match video resolution
    H, W = video.disps.shape[1:3]
    H_da3, W_da3 = da3_depth.shape

    if H_da3 != H or W_da3 != W:
        depth_resized = F.interpolate(
            depth_metric[None, None],
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )[0, 0]

        conf_resized = F.interpolate(
            da3_conf[None, None],
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )[0, 0]
    else:
        depth_resized = depth_metric
        conf_resized = da3_conf

    # Convert to disparity
    disp_da3 = 1.0 / (depth_resized + 1e-6)

    # Only use high-confidence pixels
    high_conf_mask = conf_resized > conf_threshold

    # Inject into video
    with video.get_lock():
        video.disps_sens[frame_idx][high_conf_mask] = disp_da3[high_conf_mask]
