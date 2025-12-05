# Point Cloud Misalignment Root Cause Analysis

## Problem Statement
DA3 depth maps are not properly aligned, resulting in inaccurate point cloud reconstruction.

## Root Cause

### 1. **Global Scale/Shift Problem**
Current implementation uses a **single global scale** (1.055) for all frames:
```
[Visualizer] Using DA3 depth: 61 frames, scale=1.055, shift=0.000
```

However, DA3 is **affine-invariant**, meaning each frame has its own scale/shift:
```
d_true = s_i * d_DA3_i + t_i    # Different (s_i, t_i) per frame!
```

Using global scale causes:
- Frame N with DA3 depth d₁ → metric depth = 1.055 * d₁
- Frame M with DA3 depth d₂ → metric depth = 1.055 * d₂
- But d₁ and d₂ have **different intrinsic scales** → misalignment!

### 2. **Missing Per-Frame Alignment**
Current code (`scale_consensus.py`):
- Estimates ONE global scale across all frames
- Does NOT account for per-frame affine ambiguity
- Result: Point clouds from different frames don't align

### 3. **No Geometric Consistency Enforcement**
- No reprojection error minimization
- No multi-view consistency check between DA3 depths
- DROID poses are accurate, but DA3 depths are not aligned to them

## Solution: Per-Frame Scale-Shift Alignment

### Method 1: Sparse Depth from Triangulation (Metric3D approach)
For each frame i:
1. **Triangulate sparse 3D points** using DROID poses and correlation matching
2. **Project** sparse points back to frame i → get metric depth d_metric
3. **Sample** DA3 depth at same pixels → get d_DA3
4. **Solve**: (s_i, t_i) = argmin Σ |d_metric - (s_i * d_DA3 + t_i)|²
5. **Align**: d_aligned_i = s_i * d_DA3_i + t_i

### Method 2: Depth Warping with Consistency (DepthCrafter approach)
For each frame i:
1. **Warp** DA3 depth from neighboring frames j using DROID poses
2. **Check consistency**: If |d_warped_j→i - d_DA3_i| < τ, keep pixel
3. **Fuse** consistent depths with confidence weighting
4. **Result**: Geometrically consistent depth map

### Method 3: Joint Optimization (MonoGS approach)
Optimize per-frame (s_i, t_i) jointly with poses:
- Energy: E = E_photometric + λ₁ * E_depth + λ₂ * E_consistency
- Where: E_depth = Σᵢ |d_DROID_i - (s_i * d_DA3_i + t_i)|²
- And: E_consistency = Σᵢⱼ |d_warped_j→i - d_DA3_i|²

## Recommended Implementation

**Hybrid approach combining Method 1 + 2:**

1. **Per-frame alignment** using DROID triangulation (robust, no extra optimization)
2. **Multi-view consistency filter** to remove outliers
3. **Confidence-weighted fusion** for overlapping regions

### Algorithm:
```python
for frame_idx in keyframes:
    # Step 1: Get sparse metric depth from DROID triangulation
    sparse_points_3d = triangulate_from_droid_matches(frame_idx)
    sparse_depth_metric = project_to_image(sparse_points_3d, intrinsics)

    # Step 2: Sample DA3 depth at same pixels
    da3_depth_sampled = da3_depth[frame_idx][sparse_points_2d]

    # Step 3: Robust scale-shift estimation (RANSAC + Least Squares)
    scale, shift = estimate_scale_shift_robust(
        da3_depth_sampled,
        sparse_depth_metric,
        method='ransac'  # Outlier rejection
    )

    # Step 4: Align full DA3 depth map
    da3_aligned = scale * da3_depth[frame_idx] + shift

    # Step 5: Multi-view consistency check (optional but recommended)
    for neighbor_idx in get_neighbors(frame_idx):
        depth_warped = warp_depth(
            da3_aligned[neighbor_idx],
            pose_from=poses[neighbor_idx],
            pose_to=poses[frame_idx],
            intrinsics=K
        )

        # Filter inconsistent pixels
        consistent_mask = abs(depth_warped - da3_aligned) < threshold
        da3_aligned[~consistent_mask] = 0  # Mark as invalid

    # Step 6: Store aligned depth
    video.da3_depths[frame_idx] = da3_aligned
    video.da3_scale[frame_idx] = scale  # Per-frame scale!
    video.da3_shift[frame_idx] = shift  # Per-frame shift!
```

## Key Differences from Current Code

| Current (Wrong) | Proposed (Correct) |
|----------------|-------------------|
| Global scale for all frames | Per-frame scale/shift |
| Scale from pose baseline only | Scale from triangulated sparse depth |
| No consistency check | Multi-view consistency filtering |
| Single alignment pass | Iterative refinement possible |

## Expected Improvement

- ✅ Each DA3 depth map aligned to DROID's metric space
- ✅ Geometrically consistent point clouds across frames
- ✅ Accurate 3D reconstruction
- ✅ Better visualization quality

## Implementation Priority

1. **High**: Per-frame scale-shift from DROID triangulation
2. **High**: Robust estimation (RANSAC to reject outliers)
3. **Medium**: Multi-view consistency check
4. **Low**: Iterative refinement (if initial results are good)
