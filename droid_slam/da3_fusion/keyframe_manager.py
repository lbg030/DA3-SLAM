"""
Keyframe-based DA3 Manager for efficient depth foundation model integration
"""
import torch
import torch.nn.functional as F
import threading
from collections import OrderedDict
from lietorch import SE3


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to quaternion [qw, qx, qy, qz].

    Args:
        R: 3x3 rotation matrix (torch.Tensor)

    Returns:
        q: 4D quaternion tensor [qw, qx, qy, qz]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = torch.stack([qw, qx, qy, qz])
    # Normalize to ensure unit quaternion
    q = q / torch.norm(q)

    return q


class KeyframeDA3Manager:
    """
    Manages DA3 inference on keyframes only for real-time performance.

    Key features:
    - Async inference on keyframes only (~10% of frames)
    - Caches depth, confidence, pose, and intermediate features
    - Thread-safe access to cached results
    """

    def __init__(self, da3_model, video, process_res=504):
        """
        Args:
            da3_model: DepthAnything3 model instance
            video: DepthVideo instance from DROID
            process_res: Processing resolution for DA3
        """
        self.da3 = da3_model
        self.video = video
        self.process_res = process_res

        # Cache: {frame_idx: dict with depth, conf, pose, feats}
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()

        # Track which frames are being processed
        self.processing = set()
        self.processing_lock = threading.Lock()

        # Feature layers to extract (ViT-L: 24 layers total)
        # Extract from late layers for semantic understanding
        self.feature_layers = [11, 17, 23]  # Early, mid, late features

    def should_run_da3(self, frame_idx, graph):
        """
        Determine if DA3 should run on this frame.

        Criteria:
        1. Frame is a keyframe in factor graph
        2. Not already processed or being processed
        3. Enough frames since last DA3 run

        Args:
            frame_idx: Current frame index
            graph: FactorGraph instance

        Returns:
            bool: Whether to run DA3
        """
        # Check if already cached
        with self.cache_lock:
            if frame_idx in self.cache:
                return False

        # Check if currently processing
        with self.processing_lock:
            if frame_idx in self.processing:
                return False

        # Check if it's a keyframe
        if graph.ii is None or len(graph.ii) == 0:
            return False

        is_keyframe = (frame_idx in graph.ii) or (frame_idx in graph.jj)

        return is_keyframe

    def run_da3_async(self, frame_idx):
        """
        Run DA3 inference asynchronously in a separate thread.

        Args:
            frame_idx: Frame index to process
        """
        thread = threading.Thread(
            target=self._run_da3_sync,
            args=(frame_idx,),
            daemon=True
        )
        thread.start()

    def _run_da3_sync(self, frame_idx):
        """
        Actual DA3 inference (runs in separate thread).

        Args:
            frame_idx: Frame index to process
        """
        # Mark as processing
        with self.processing_lock:
            if frame_idx in self.processing:
                return
            self.processing.add(frame_idx)

        try:
            # Get image from video buffer
            with self.video.get_lock():
                image = self.video.images[frame_idx].clone()  # [3, H, W]

            # Prepare for DA3 (expects [B, N, 3, H, W])
            H, W = image.shape[-2:]
            image_input = image[None, None].float()  # [1, 1, 3, H, W]

            # Resize to process_res if needed
            if H != self.process_res or W != self.process_res:
                image_input = F.interpolate(
                    image_input.squeeze(1),  # [1, 3, H, W]
                    size=(self.process_res, self.process_res),
                    mode='bilinear',
                    align_corners=False
                ).unsqueeze(1)  # [1, 1, 3, process_res, process_res]

            # Run DA3 inference
            with torch.no_grad():
                output = self.da3.forward(
                    image_input,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=self.feature_layers,
                    infer_gs=False  # Don't need 3DGS
                )

            # Extract depth and confidence
            depth_raw = output['depth'][0, 0]  # [process_res, process_res]
            conf_raw = output.get('depth_conf', None)
            if conf_raw is not None:
                conf_raw = conf_raw[0, 0]  # [process_res, process_res]
            else:
                # If no confidence, use uniform
                conf_raw = torch.ones_like(depth_raw)

            # Resize back to original resolution
            depth = F.interpolate(
                depth_raw[None, None],
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )[0, 0]

            conf = F.interpolate(
                conf_raw[None, None],
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )[0, 0]

            # Extract pose if available
            pose_da3 = None
            if 'extrinsics' in output and output['extrinsics'] is not None:
                # DA3 may return either a 4x4 homogeneous extrinsic or a 3x4 [R|t].
                # Normalize to a 4x4 matrix before inversion.
                ext = output['extrinsics'][0, 0]

                # If DA3 returned a 3x4 [R|t], convert to 4x4 homogeneous matrix.
                if ext.ndim == 2 and ext.shape[0] == 3 and ext.shape[1] == 4:
                    ext4 = torch.eye(4, device=ext.device, dtype=ext.dtype)
                    ext4[:3, :4] = ext
                    ext = ext4

                # If shape is now 4x4, invert to get camera-to-world (c2w).
                if ext.ndim == 2 and ext.shape[0] == 4 and ext.shape[1] == 4:
                    c2w = torch.linalg.inv(ext)
                    # Extract rotation and translation
                    R = c2w[:3, :3]
                    t = c2w[:3, 3]
                    # Convert rotation matrix to quaternion
                    q = rotation_matrix_to_quaternion(R)
                    # SE3 format: [tx, ty, tz, qw, qx, qy, qz]
                    pose_da3 = torch.cat([t, q])
                else:
                    # Unexpected shape: skip pose extraction
                    raise ValueError(f"Unexpected extrinsics shape from DA3: {ext.shape}")
                    

            # Extract intermediate features
            feats = {}
            if 'aux' in output and output['aux'] is not None:
                for key, val in output['aux'].items():
                    if key.startswith('feat_layer_'):
                        # Features shape: [1, 1, H/14, W/14, C]
                        feat = val[0, 0]  # [H/14, W/14, C]
                        feats[key] = feat

            # Store in cache
            with self.cache_lock:
                self.cache[frame_idx] = {
                    'depth': depth.cpu(),  # Move to CPU to save GPU memory
                    'conf': conf.cpu(),
                    'pose': pose_da3.cpu() if pose_da3 is not None else None,
                    'feats': {k: v.cpu() for k, v in feats.items()},
                    'resolution': (H, W)
                }

                # Limit cache size to prevent memory overflow
                max_cache_size = 100
                if len(self.cache) > max_cache_size:
                    # Remove oldest entry
                    self.cache.popitem(last=False)

        finally:
            # Remove from processing set
            with self.processing_lock:
                self.processing.discard(frame_idx)

    def get_data(self, frame_idx):
        """
        Get cached DA3 data for a frame.

        Args:
            frame_idx: Frame index

        Returns:
            dict or None: Cached data if available
        """
        with self.cache_lock:
            if frame_idx in self.cache:
                # Move data back to GPU
                data = self.cache[frame_idx]
                return {
                    'depth': data['depth'].cuda(),
                    'conf': data['conf'].cuda(),
                    'pose': data['pose'].cuda() if data['pose'] is not None else None,
                    'feats': {k: v.cuda() for k, v in data['feats'].items()},
                    'resolution': data['resolution']
                }
        return None

    def has_data(self, frame_idx):
        """Check if data is available for a frame."""
        with self.cache_lock:
            return frame_idx in self.cache

    def clear_old_data(self, before_idx):
        """Remove cached data older than specified index."""
        with self.cache_lock:
            keys_to_remove = [k for k in self.cache.keys() if k < before_idx]
            for k in keys_to_remove:
                del self.cache[k]
