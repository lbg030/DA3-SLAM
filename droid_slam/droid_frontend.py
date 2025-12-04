import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph

from cuda_timer import CudaTimer

# DA3 Fusion imports
from da3_fusion import (
    KeyframeDA3Manager,
    MultiViewScaleConsensus,
    DepthGuidedPoseRefiner,
    inject_depth_prior_to_video,
    DA3FeatureMatcher
)


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

        # DA3 Fusion components
        self.use_da3_fusion = da3_model is not None and hasattr(args, 'use_da3_fusion') and args.use_da3_fusion
        if self.use_da3_fusion:
            print("[DA3-Fusion] Initializing DA3 fusion components...")
            self.da3_manager = KeyframeDA3Manager(da3_model, video)
            self.scale_consensus = MultiViewScaleConsensus()
            self.pose_refiner = DepthGuidedPoseRefiner(video)
            self.feature_matcher = DA3FeatureMatcher()
            self.current_scale = 1.0
            print("[DA3-Fusion] Initialization complete!")

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
        """Process available DA3 data and fuse with DROID"""
        # Collect frames with DA3 data
        available_frames = []
        for idx in range(max(0, self.t1 - 20), self.t1):
            if self.da3_manager.has_data(idx):
                available_frames.append(idx)

        if len(available_frames) < 3:
            return  # Need at least 3 frames for scale consensus

        # Get recent frames for scale estimation
        recent_frames = available_frames[-min(5, len(available_frames)):]

        try:
            # Extract poses
            da3_poses = []
            droid_poses = []
            for idx in recent_frames:
                da3_data = self.da3_manager.get_data(idx)
                if da3_data is not None and da3_data['pose'] is not None:
                    da3_poses.append(da3_data['pose'])
                    droid_poses.append(self.video.poses[idx])

            # Estimate scale if we have DA3 poses
            if len(da3_poses) >= 2:
                self.current_scale = self.scale_consensus.estimate_scale(
                    da3_poses, droid_poses, recent_frames
                )

            # Inject depth priors for available frames
            for idx in available_frames:
                da3_data = self.da3_manager.get_data(idx)
                if da3_data is not None:
                    try:
                        # Inject high-confidence depth as prior
                        inject_depth_prior_to_video(
                            self.video,
                            idx,
                            da3_data['depth'],
                            da3_data['conf'],
                            self.current_scale,
                            conf_threshold=0.7
                        )

                        # Add DA3 features for loop closure
                        if len(da3_data['feats']) > 0:
                            self.feature_matcher.add_features(idx, da3_data['feats'])

                    except Exception as e:
                        print(f"[DA3-Fusion] Warning: Failed to inject depth for frame {idx}: {e}")

            # Loop closure detection (every 10 frames)
            if self.count % 10 == 0 and len(available_frames) > 0:
                current_idx = available_frames[-1]
                loop_candidates = self.feature_matcher.find_loop_closures(current_idx)

                if len(loop_candidates) > 0:
                    print(f"[DA3-Fusion] Found {len(loop_candidates)} loop closure candidates for frame {current_idx}")
                    # Note: Adding loop edges directly in factor graph can be unstable
                    # For now, just detect and log. Can be enabled for backend optimization.

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
