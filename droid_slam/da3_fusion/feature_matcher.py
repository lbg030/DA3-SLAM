"""
Loop closure detection using DA3's DINOv2 features
"""
import torch
import torch.nn.functional as F


class DA3FeatureMatcher:
    """
    Uses DA3's intermediate DINOv2 features for loop closure detection.

    Key advantages over DROID's features:
    - Semantic understanding from DINOv2 pretraining
    - Finer spatial resolution (14x14 vs 8x8)
    - Better for long-term matching
    """

    def __init__(self, similarity_threshold=0.85, temporal_gap=30):
        """
        Args:
            similarity_threshold: Minimum cosine similarity for loop closure
            temporal_gap: Minimum frame gap to consider loop closure
        """
        self.feature_db = {}  # {frame_idx: {'global': desc, 'spatial': feat}}
        self.similarity_threshold = similarity_threshold
        self.temporal_gap = temporal_gap

    def add_features(self, frame_idx, da3_features):
        """
        Add DA3 features to database.

        Args:
            frame_idx: Frame index
            da3_features: Dict of features from DA3 (keys: 'feat_layer_X')
        """
        if len(da3_features) == 0:
            return

        # Use the last layer features (most semantic)
        # Find the highest layer number
        layer_keys = [k for k in da3_features.keys() if k.startswith('feat_layer_')]
        if len(layer_keys) == 0:
            return

        # Sort by layer number and get the last one
        layer_nums = [int(k.split('_')[-1]) for k in layer_keys]
        max_layer_idx = layer_nums.index(max(layer_nums))
        feat_key = layer_keys[max_layer_idx]

        feat_spatial = da3_features[feat_key]  # [H/14, W/14, C]

        # Compute global descriptor (for fast retrieval)
        global_desc = feat_spatial.mean(dim=[0, 1])  # [C]
        global_desc = F.normalize(global_desc, dim=0)

        # Store both
        self.feature_db[frame_idx] = {
            'global': global_desc,
            'spatial': feat_spatial
        }

    def find_loop_closures(self, current_idx, max_candidates=5):
        """
        Find potential loop closures for current frame.

        Args:
            current_idx: Current frame index
            max_candidates: Maximum number of candidates to return

        Returns:
            List of (frame_idx, similarity_score) tuples
        """
        if current_idx not in self.feature_db:
            return []

        current_desc = self.feature_db[current_idx]['global']
        candidates = []

        # Search in temporal window
        search_start = max(0, current_idx - 200)
        search_end = current_idx - self.temporal_gap

        for past_idx in range(search_start, search_end):
            if past_idx in self.feature_db:
                past_desc = self.feature_db[past_idx]['global']

                # Cosine similarity
                similarity = (current_desc * past_desc).sum().item()

                if similarity > self.similarity_threshold:
                    candidates.append((past_idx, similarity))

        # Sort by similarity and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]

    def compute_dense_correspondence(self, frame_i, frame_j, radius=3):
        """
        Compute dense correspondence between two frames using DA3 features.

        Args:
            frame_i, frame_j: Frame indices
            radius: Correlation radius

        Returns:
            coords_ij: Predicted coordinates in j from i [H, W, 2]
            confidence: Matching confidence [H, W]
        """
        if frame_i not in self.feature_db or frame_j not in self.feature_db:
            return None, None

        feat_i = self.feature_db[frame_i]['spatial']  # [H, W, C]
        feat_j = self.feature_db[frame_j]['spatial']

        H, W, C = feat_i.shape

        # Normalize features
        feat_i_norm = F.normalize(feat_i, dim=-1)
        feat_j_norm = F.normalize(feat_j, dim=-1)

        # Compute correlation volume (memory efficient)
        # Reshape for matrix multiplication
        feat_i_flat = feat_i_norm.reshape(-1, C)  # [H*W, C]
        feat_j_flat = feat_j_norm.reshape(-1, C)  # [H*W, C]

        # Correlation: [H*W, H*W]
        # Too large! Use local correlation instead

        # Create coordinate grid
        coords_i = torch.stack(torch.meshgrid(
            torch.arange(H, device=feat_i.device),
            torch.arange(W, device=feat_i.device),
            indexing='ij'
        ), dim=-1).float()  # [H, W, 2]

        coords_j = torch.zeros_like(coords_i)
        confidence = torch.zeros(H, W, device=feat_i.device)

        # For each position in i, find best match in local window of j
        for h in range(H):
            for w in range(W):
                # Feature at (h, w) in i
                f_i = feat_i_norm[h, w]  # [C]

                # Search window in j
                h_start = max(0, h - radius)
                h_end = min(H, h + radius + 1)
                w_start = max(0, w - radius)
                w_end = min(W, w + radius + 1)

                # Features in search window
                f_j_window = feat_j_norm[h_start:h_end, w_start:w_end]  # [h_size, w_size, C]

                # Compute similarities
                sim = (f_j_window * f_i[None, None, :]).sum(dim=-1)  # [h_size, w_size]

                # Find best match
                max_idx = sim.argmax()
                max_h, max_w = max_idx // sim.shape[1], max_idx % sim.shape[1]

                # Convert to global coordinates
                coords_j[h, w, 0] = h_start + max_h
                coords_j[h, w, 1] = w_start + max_w
                confidence[h, w] = sim[max_h, max_w]

        return coords_j, confidence

    def add_loop_edges_to_graph(self, graph, loop_candidates, coords0):
        """
        Add loop closure edges to DROID factor graph.

        Args:
            graph: FactorGraph instance
            loop_candidates: List of (frame_i, frame_j, similarity) tuples
            coords0: Coordinate grid [H, W, 2]

        Returns:
            Number of edges added
        """
        num_added = 0

        for frame_i, frame_j, similarity in loop_candidates:
            # Check if edge already exists
            existing = ((graph.ii == frame_i) & (graph.jj == frame_j)).any()
            if existing:
                continue

            # Compute dense correspondence
            coords_j, conf = self.compute_dense_correspondence(
                frame_i, frame_j, radius=5
            )

            if coords_j is None:
                continue

            # Resize to match graph resolution
            H_graph, W_graph = coords0.shape[:2]
            H_feat, W_feat = coords_j.shape[:2]

            if H_feat != H_graph or W_feat != W_graph:
                # Resize using bilinear interpolation
                # Normalize coordinates to [-1, 1]
                coords_j_norm = coords_j.clone()
                coords_j_norm[:, :, 0] = coords_j_norm[:, :, 0] / (H_feat - 1) * (H_graph - 1)
                coords_j_norm[:, :, 1] = coords_j_norm[:, :, 1] / (W_feat - 1) * (W_graph - 1)

                coords_resized = F.interpolate(
                    coords_j_norm.permute(2, 0, 1)[None],  # [1, 2, H, W]
                    size=(H_graph, W_graph),
                    mode='bilinear',
                    align_corners=False
                )[0].permute(1, 2, 0)  # [H_graph, W_graph, 2]

                conf_resized = F.interpolate(
                    conf[None, None],
                    size=(H_graph, W_graph),
                    mode='bilinear',
                    align_corners=False
                )[0, 0]  # [H_graph, W_graph]
            else:
                coords_resized = coords_j
                conf_resized = conf

            # Add edge to graph
            try:
                ii_new = torch.tensor([frame_i], device=graph.ii.device, dtype=torch.long)
                jj_new = torch.tensor([frame_j], device=graph.ii.device, dtype=torch.long)

                graph.add_factors(ii_new, jj_new, remove=False)

                # Set target and weight
                # Target is the predicted coordinates
                target = coords_resized[None]  # [1, H, W, 2]
                weight = conf_resized[None, :, :, None].repeat(1, 1, 1, 2)  # [1, H, W, 2]

                # Scale weight by similarity
                weight = weight * similarity * 0.5  # Reduce weight for loop closures

                # Append to graph targets
                graph.target = torch.cat([graph.target, target], dim=1)
                graph.weight = torch.cat([graph.weight, weight], dim=1)

                num_added += 1

            except Exception as e:
                print(f"Warning: Failed to add loop edge {frame_i}->{frame_j}: {e}")
                continue

        return num_added

    def clear_old_features(self, before_idx):
        """Remove features older than specified index."""
        keys_to_remove = [k for k in self.feature_db.keys() if k < before_idx]
        for k in keys_to_remove:
            del self.feature_db[k]
