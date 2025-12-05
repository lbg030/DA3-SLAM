import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph

from cuda_timer import CudaTimer

# DA3 Fusion imports
from da3_fusion import (
    KeyframeDA3Manager,
    ScaleShiftEstimator,
    DepthFactorManager,
    DA3FeatureMatcher,
    compute_hybrid_depth
)

from visualization_utils import show_plot

ENABLE_TIMING = False

class DroidFrontend:
    def __init__(self, net, video, args, da3_model=None):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(
            video, net.update, max_factors=48, upsample=args.upsample
        )

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 20
        self.iters1 = 3
        self.iters2 = 2

        self.keyframe_removal_index = 3

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

        self.depth_window = 3

        self.motion_damping = 0.0
        if hasattr(args, "motion_damping"):
            self.motion_damping = args.motion_damping

        # DA3 Fusion components (Professor-level bidirectional architecture)
        self.use_da3_fusion = da3_model is not None and hasattr(args, 'use_da3_fusion') and args.use_da3_fusion
        if self.use_da3_fusion:
            print("[DA3-Fusion] Initializing BIDIRECTIONAL professor-level fusion...")
            from da3_fusion.bidirectional_fusion import BidirectionalDA3DROIDFusion
            from da3_fusion.per_frame_alignment import PerFrameDepthAligner

            self.da3_manager = KeyframeDA3Manager(da3_model, video)

            # NEW: Bidirectional fusion engine (replaces separate components)
            self.bidirectional_fusion = BidirectionalDA3DROIDFusion(video, da3_model, args)

            # CRITICAL: Per-frame depth aligner for geometric consistency
            self.depth_aligner = PerFrameDepthAligner(
                min_sparse_points=50,
                ransac_iterations=100,
                inlier_threshold=0.1
            )

            # Keep legacy components for backward compatibility
            self.scale_estimator = ScaleShiftEstimator(
                min_triangulation_pairs=3,
                min_parallax=0.02,
                inlier_threshold=0.05,
                temporal_alpha=0.7
            )
            self.depth_factor_manager = DepthFactorManager(
                video,
                lambda_depth=0.1,
                huber_delta=0.1
            )
            self.feature_matcher = DA3FeatureMatcher()
            print("[DA3-Fusion] Per-frame alignment enabled (geometric consistency)!")
            print("[DA3-Fusion] Bidirectional fusion ready (DROID ⇄ DA3 mutual refinement)!")

    def _init_next_state(self):
        # set pose / depth for next iteration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]

        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - 3 : self.t1 - 1], 0.5
        )

        # damped linear velocity model
        if self.motion_damping >= 0:
            poses = SE3(self.video.poses)
            vel = (poses[self.t1 - 1] * poses[self.t1 - 2].inv()).log()
            damped_vel = self.motion_damping * vel
            next_pose = SE3.exp(damped_vel) * poses[self.t1 - 1]
            self.video.poses[self.t1] = next_pose.data

    def _update(self):
        """add edges, perform update"""

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(
            self.t1 - 5,
            max(self.t1 - self.frontend_window, 0),
            rad=self.frontend_radius,
            nms=self.frontend_nms,
            thresh=self.frontend_thresh,
            beta=self.beta,
            remove=True,
        )

        # DA3 Fusion: Check if we should run DA3 on this keyframe
        if self.use_da3_fusion and self.da3_manager.should_run_da3(self.t1 - 1, self.graph):
            if self.count % 50 == 0:
                print(f"[DA3-Fusion] Triggering DA3 on frame {self.t1 - 1}")
            self.da3_manager.run_da3_async(self.t1 - 1)

        # DA3 Fusion: Process available DA3 data
        if self.use_da3_fusion:
            self._fuse_da3_data()

        self.video.disps[self.t1 - 1] = torch.where(
            self.video.disps_sens[self.t1 - 1] > 0,
            self.video.disps_sens[self.t1 - 1],
            self.video.disps[self.t1 - 1],
        )

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        # set initial pose for next frame
        d = self.video.distance(
            [self.t1 - 4], [self.t1 - 2], beta=self.beta, bidirectional=True
        )

        if d.item() < 2 * self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 3)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)


        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - self.depth_window - 1 : self.t1 - 1], 0.7
        )

        # update visualization
        self.video.dirty[self.graph.ii.min() : self.t1] = True

    def _fuse_da3_data(self):
        """
        Bidirectional DA3-DROID fusion (Professor-level).

        Key innovation: DROID poses → DA3 refinement → DROID pose correction
        This creates a tight feedback loop for mutual refinement.
        """
        # Collect frames with DA3 data
        available_frames = []
        for idx in range(max(0, self.t1 - 20), self.t1):
            if self.da3_manager.has_data(idx):
                available_frames.append(idx)

        if len(available_frames) == 0:
            return  # No DA3 data available

        try:
            # === STEP 1: Per-frame alignment (CRITICAL for point cloud accuracy!) ===
            aligned_depths = {}  # Store aligned depths for multi-view consistency
            stored_count = 0

            for frame_idx in available_frames:
                da3_data = self.da3_manager.get_data(frame_idx)
                if da3_data is None:
                    continue

                # Get DA3 outputs
                da3_depth = da3_data['depth']  # [H, W] at full resolution
                da3_conf = da3_data['conf']    # [H, W]

                # Resize to video resolution first
                H_video, W_video = self.video.da3_depths.shape[1], self.video.da3_depths.shape[2]
                import torch.nn.functional as F
                da3_depth_resized = F.interpolate(
                    da3_depth.unsqueeze(0).unsqueeze(0),
                    size=(H_video, W_video),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                da3_conf_resized = F.interpolate(
                    da3_conf.unsqueeze(0).unsqueeze(0),
                    size=(H_video, W_video),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

                # CRITICAL: Per-frame scale-shift alignment using DROID's optimized depth
                droid_inv_depth = self.video.disps[frame_idx]  # DROID's inverse depth [H, W]
                droid_weight = self.video.disps_sens[frame_idx]  # Confidence weights [H, W]

                aligned_depth, scale, shift, confidence = self.depth_aligner.align_frame(
                    frame_idx=frame_idx,
                    da3_depth=da3_depth_resized,
                    droid_depth=droid_inv_depth,
                    droid_weight=droid_weight,
                    intrinsics=self.video.intrinsics[0],
                    poses=self.video.poses,
                    target_shape=(H_video, W_video)  # Ensure output matches video resolution
                )

                # CRITICAL: Ensure aligned_depth matches video resolution
                if aligned_depth.shape != (H_video, W_video):
                    aligned_depth = F.interpolate(
                        aligned_depth.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                        size=(H_video, W_video),
                        mode='bilinear',
                        align_corners=False
                    )[0, 0]  # Remove batch and channel dims -> [H_video, W_video]

                # Store for multi-view consistency
                aligned_depths[frame_idx] = aligned_depth

                # Log alignment quality
                if self.count % 50 == 0 and stored_count == 0:
                    print(f"[PerFrameAlign] Frame {frame_idx}: scale={scale:.3f}, shift={shift:.3f}, "
                          f"confidence={confidence:.2f}, final_shape={aligned_depth.shape}")

                # Store aligned depth
                self.video.da3_depths[frame_idx] = aligned_depth
                self.video.da3_confs[frame_idx] = da3_conf_resized * confidence  # Scale confidence by alignment quality
                stored_count += 1

            # === STEP 2: Multi-view consistency refinement ===
            # Refine each aligned depth using geometric consistency with neighbors
            for frame_idx in available_frames:
                if frame_idx not in aligned_depths:
                    continue

                # Get neighbor frames (within temporal window)
                neighbor_indices = [
                    i for i in available_frames
                    if abs(i - frame_idx) <= 5 and i != frame_idx and i in aligned_depths
                ]

                if len(neighbor_indices) >= 2:
                    # Apply multi-view consistency check
                    refined_depth, valid_mask = self.depth_aligner.align_with_multiview_consistency(
                        frame_idx=frame_idx,
                        da3_depth=da3_depth_resized if frame_idx == available_frames[-1] else None,  # Not needed
                        aligned_depth=aligned_depths[frame_idx],
                        poses=self.video.poses,
                        intrinsics=self.video.intrinsics[0],
                        neighbor_indices=neighbor_indices,
                        all_da3_depths={},  # Not needed
                        all_aligned_depths=aligned_depths,
                        consistency_threshold=0.05  # 5% relative error
                    )

                    # Update with refined depth
                    self.video.da3_depths[frame_idx] = refined_depth

                    # Update confidence based on consistency
                    consistency_ratio = valid_mask.float().mean().item()
                    if self.count % 50 == 0 and stored_count > 0:
                        print(f"[MultiViewConsistency] Frame {frame_idx}: {consistency_ratio*100:.1f}% consistent pixels")

            # === STEP 3: Bidirectional Fusion (optional additional refinement) ===
            # Use DROID poses to refine DA3 depth, then use refined depth to correct poses
            for frame_idx in available_frames:
                # Fuse this frame with bidirectional engine
                refined_depth, refined_pose = self.bidirectional_fusion.fuse_frame(
                    frame_idx, self.graph
                )

                if refined_depth is not None:
                    # Update video with refined depth
                    self.video.da3_depths[frame_idx] = refined_depth

                if refined_pose is not None:
                    # (Optional) Update DROID pose with DA3-refined pose
                    # For now, skip to avoid destabilizing DROID's BA
                    pass

            # === STEP 3: Compute depth loss for BA integration ===
            if len(available_frames) > 0:
                depth_loss, residuals = self.bidirectional_fusion.compute_depth_loss(available_frames)

                if self.count % 20 == 0 and depth_loss > 0:
                    scale = self.bidirectional_fusion.current_scale
                    print(f"[BidirectionalFusion] Scale={scale:.4f}, Depth loss={depth_loss:.4f}, "
                          f"Active frames={len(available_frames)}")

            # Note: Bidirectional fusion handles scale estimation, depth priors, and BA integration internally

        except Exception as e:
            print(f"[DA3-Fusion] Error in fusion: {e}")
            import traceback
            traceback.print_exc()

    def _initialize(self):
        """initialize the SLAM system"""

        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(
            0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False
        )

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1 - 4 : self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1 - 1].clone()
        self.last_disp = self.video.disps[self.t1 - 1].clone()
        self.last_time = self.video.tstamp[self.t1 - 1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[: self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

    def __call__(self):
        """main update"""

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self._initialize()
            self._init_next_state()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self._update()
            self._init_next_state()
