"""
DA3-SLAM Fusion Module

Professor-level integration of Depth Anything V3 with DROID-SLAM.

Key Components:
    - KeyframeDA3Manager: Asynchronous DA3 inference on keyframes
    - ScaleShiftEstimator: Geometric scale-shift alignment with triangulation validation
    - DepthFactorManager: Direct depth factor integration into bundle adjustment
    - DA3FeatureMatcher: Loop closure detection using DINOv2 features

Architecture:
    1. DA3 runs on ~10% of frames (keyframes only)
    2. Scale-shift estimated via multi-view triangulation
    3. Depth factors added to factor graph with confidence weighting
    4. Hybrid depth map for visualization (DROID + DA3 fusion)
"""

from .keyframe_manager import KeyframeDA3Manager
from .geometric_alignment import ScaleShiftEstimator, triangulate_points, umeyama_alignment
from .depth_factors import DepthFactorManager, compute_hybrid_depth
from .feature_matcher import DA3FeatureMatcher
from .da3_projection import iproj_depth, depth_filter_multiview, create_point_cloud_from_da3, estimate_scale_from_poses_and_depths
from .bidirectional_fusion import BidirectionalDA3DROIDFusion
from .multiview_da3 import MultiViewDA3Inference
from .per_frame_alignment import PerFrameDepthAligner

# Legacy imports for backward compatibility
from .scale_consensus import MultiViewScaleConsensus, align_depth_to_metric_scale, compute_disparity_from_depth
from .pose_refinement import DepthGuidedPoseRefiner, inject_depth_prior_to_video

__all__ = [
    'KeyframeDA3Manager',
    'ScaleShiftEstimator',
    'DepthFactorManager',
    'DA3FeatureMatcher',
    'BidirectionalDA3DROIDFusion',
    'MultiViewDA3Inference',
    'PerFrameDepthAligner',
    'triangulate_points',
    'umeyama_alignment',
    'compute_hybrid_depth',
    'iproj_depth',
    'depth_filter_multiview',
    'create_point_cloud_from_da3',
    'estimate_scale_from_poses_and_depths',
    # Legacy
    'MultiViewScaleConsensus',
    'align_depth_to_metric_scale',
    'compute_disparity_from_depth',
    'DepthGuidedPoseRefiner',
    'inject_depth_prior_to_video',
]
