"""
DA3 Fusion Module for DROID-SLAM

Integrates Depth Anything V3 with DROID-SLAM for improved performance.
"""

from .keyframe_manager import KeyframeDA3Manager
from .scale_consensus import MultiViewScaleConsensus, align_depth_to_metric_scale, compute_disparity_from_depth
from .pose_refinement import DepthGuidedPoseRefiner, inject_depth_prior_to_video
from .feature_matcher import DA3FeatureMatcher

__all__ = [
    'KeyframeDA3Manager',
    'MultiViewScaleConsensus',
    'align_depth_to_metric_scale',
    'compute_disparity_from_depth',
    'DepthGuidedPoseRefiner',
    'inject_depth_prior_to_video',
    'DA3FeatureMatcher',
]
