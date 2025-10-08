import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphLayer(nn.Module):
    """
    Custom graph layer combining message passing with adaptive edge weighting
    """

    def __init__(self, in_features, out_features, num_heads=4, dropout=0.3):
        super(AdaptiveGraphLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads

        # Multi-head transformations
        self.W_query = nn.Linear(in_features, out_features)
        self.W_key = nn.Linear(in_features, out_features)
        self.W_value = nn.Linear(in_features, out_features)

        # Edge feature integration (for flow-based graphs)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_features * 2, out_features), nn.ReLU(), nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x, edge_index, edge_attr=None):
        """
        x: Node features [num_nodes, in_features]
        edge_index: Graph connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, in_features] (optional)
        """
        batch_size, num_nodes, in_features = x.shape

        # Reshape for multi-head attention
        Q = self.W_query(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.W_key(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.W_value(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, nodes, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply edge mask if provided
        if edge_index is not None:
            mask = torch.zeros(
                batch_size, self.num_heads, num_nodes, num_nodes, device=x.device
            )
            mask[:, :, edge_index[0], edge_index[1]] = 1
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate
        out = torch.matmul(attn_weights, V)  # [batch, heads, nodes, head_dim]
        out = out.transpose(1, 2).contiguous()  # [batch, nodes, heads, head_dim]
        out = out.view(batch_size, num_nodes, self.out_features)

        # Residual connection + layer norm
        out = self.layer_norm(out + x if x.shape[-1] == self.out_features else out)

        return out, attn_weights


# class HybridAGNN_IDS(nn.Module):
#     """
#     Hybrid Adaptive Graph Neural Network for Intrusion Detection
#     Combines graph convolution with temporal features for IDS datasets
#     """

#     def __init__(
#         self,
#         num_classes,
#         input_dim=4,
#         hidden_dims=[128, 256, 128],
#         num_heads=4,
#         dropout_rate=0.4,
#         graph_layers=3,
#     ):
#         super(HybridAGNN_IDS, self).__init__()

#         self.num_classes = num_classes

#         # Initial feature transformation
#         self.feature_encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dims[0]),
#             nn.BatchNorm1d(hidden_dims[0]),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#         )

#         # Graph layers with varying dimensions
#         self.graph_layers = nn.ModuleList()
#         dims = [hidden_dims[0]] + hidden_dims

#         for i in range(graph_layers):
#             self.graph_layers.append(
#                 AdaptiveGraphLayer(
#                     dims[i], dims[i + 1], num_heads=num_heads, dropout=dropout_rate
#                 )
#             )

#         # Global pooling strategies (combine multiple aggregations)
#         self.pool_attention = nn.Linear(dims[-1], 1)

#         # Classifier head with bottleneck
#         self.classifier = nn.Sequential(
#             nn.Linear(dims[-1] * 2, dims[-1]),  # *2 for concatenated pooling
#             nn.BatchNorm1d(dims[-1]),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(dims[-1], dims[-1] // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate * 0.5),
#             nn.Linear(dims[-1] // 2, num_classes),
#         )

#         self.output_act = nn.Sigmoid() if num_classes == 1 else nn.Identity()

#     def global_pooling(self, x):
#         """
#         Combine mean and attention-weighted pooling
#         x: [batch, num_nodes, features]
#         """
#         # Mean pooling
#         mean_pool = torch.mean(x, dim=1)

#         # Attention pooling
#         attn_scores = self.pool_attention(x)  # [batch, num_nodes, 1]
#         attn_weights = F.softmax(attn_scores, dim=1)
#         attn_pool = torch.sum(x * attn_weights, dim=1)  # [batch, features]

#         # Concatenate both
#         return torch.cat([mean_pool, attn_pool], dim=-1)

#     def forward(self, x, edge_index=None):
#         """
#         x: Input features [batch, num_nodes, input_dim] or [batch, input_dim]
#         edge_index: Graph edges [2, num_edges] (optional, can be constructed dynamically)
#         """
#         batch_size = x.shape[0]

#         # Handle flat input (convert to graph nodes)
#         if x.dim() == 2:
#             # Reshape into pseudo-nodes (e.g., feature groups)
#             num_features = x.shape[1]
#             num_nodes = min(num_features, 20)  # Cap at 20 nodes for efficiency
#             node_dim = num_features // num_nodes

#             # Reshape and pad if necessary
#             if num_features % num_nodes != 0:
#                 padding = num_nodes - (num_features % num_nodes)
#                 x = F.pad(x, (0, padding))
#                 node_dim = x.shape[1] // num_nodes

#             x = x.view(batch_size, num_nodes, node_dim)

#         # Encode features
#         batch_size, num_nodes, _ = x.shape
#         x_flat = x.view(batch_size * num_nodes, -1)
#         x_encoded = self.feature_encoder(x_flat)
#         x = x_encoded.view(batch_size, num_nodes, -1)

#         # Construct edge_index if not provided (fully connected within each sample)
#         if edge_index is None:
#             # Create a fully connected graph for each batch sample
#             row = torch.arange(num_nodes, device=x.device).repeat_interleave(num_nodes)
#             col = torch.arange(num_nodes, device=x.device).repeat(num_nodes)
#             edge_index = torch.stack([row, col], dim=0)

#         # Apply graph layers
#         for graph_layer in self.graph_layers:
#             x, _ = graph_layer(x, edge_index)

#         # Global pooling
#         x_pooled = self.global_pooling(x)

#         # Classification
#         out = self.classifier(x_pooled)
#         return self.output_act(out)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridAGNN_IDS(nn.Module):
    """
    Hybrid Adaptive Graph Neural Network for Intrusion Detection
    Combines graph convolution with temporal features for IDS datasets
    """

    def __init__(
        self,
        num_classes,
        input_dim=4,  # kept for API compatibility; not used for sizing
        hidden_dims=[128, 256, 128],
        num_heads=4,
        dropout_rate=0.4,
        graph_layers=3,
        max_auto_nodes=20,  # cap for auto node splitting when x is flat
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.graph_layers_count = graph_layers
        self.max_auto_nodes = max_auto_nodes

        # feature encoder is created lazily once we know node_dim
        self.feature_encoder = None

        # Graph layers with varying dimensions (based on hidden_dims)
        # dims = [in0, out0, out1, ..., out_{L-1}] where in0 = hidden_dims[0]
        dims = [hidden_dims[0]] + hidden_dims
        self.graph_layers = nn.ModuleList(
            [
                AdaptiveGraphLayer(
                    dims[i], dims[i + 1], num_heads=num_heads, dropout=dropout_rate
                )
                for i in range(graph_layers)
            ]
        )

        # Global pooling
        self.pool_attention = nn.Linear(hidden_dims[-1], 1)

        # Classifier head with bottleneck
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dims[-1] * 2, hidden_dims[-1]
            ),  # *2 for mean + attn pool concat
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dims[-1] // 2, num_classes),
        )

        self.output_act = nn.Sigmoid() if num_classes == 1 else nn.Identity()

    # --- lazy creation of the feature encoder, on the correct device ---
    def _ensure_feature_encoder(self, node_dim: int, device):
        if self.feature_encoder is None:
            self.feature_encoder = nn.Sequential(
                nn.Linear(node_dim, self.hidden_dims[0]),
                nn.BatchNorm1d(self.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
            ).to(device)

    def global_pooling(self, x):
        """
        Combine mean and attention-weighted pooling
        x: [batch, num_nodes, features]
        """
        mean_pool = torch.mean(x, dim=1)  # [B, F]
        attn_scores = self.pool_attention(x)  # [B, N, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N, 1]
        attn_pool = torch.sum(x * attn_weights, dim=1)  # [B, F]
        return torch.cat([mean_pool, attn_pool], dim=-1)  # [B, 2F]

    def forward(self, x, edge_index=None):
        """
        x: [batch, num_nodes, node_dim]  OR  [batch, flat_dim]
        edge_index: [2, num_edges] (shared for the batch)
        """
        B = x.shape[0]
        device = x.device

        # If x is flat, split into pseudo-nodes deterministically
        if x.dim() == 2:
            F_flat = x.shape[1]
            num_nodes = min(F_flat, self.max_auto_nodes)
            # use ceil to avoid node_dim=0 and to match deterministic partitioning
            node_dim = math.ceil(F_flat / num_nodes)
            pad = num_nodes * node_dim - F_flat
            if pad > 0:
                x = F.pad(x, (0, pad))
            x = x.view(B, num_nodes, node_dim)
        else:
            # already [B, N, D]
            num_nodes = x.shape[1]
            node_dim = x.shape[2]

        # Ensure encoder exists and is on the right device
        self._ensure_feature_encoder(node_dim, device)

        # Encode per-node features
        x_flat = x.view(B * num_nodes, node_dim)  # [B*N, D]
        x_encoded = self.feature_encoder(x_flat)  # -> hidden_dims[0]
        x = x_encoded.view(B, num_nodes, -1)  # [B, N, hidden_dims[0]]

        # Construct fully-connected edges if needed (on correct device)
        if edge_index is None:
            row = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
            col = torch.arange(num_nodes, device=device).repeat(num_nodes)
            edge_index = torch.stack([row, col], dim=0)  # [2, N*N]

        # Graph propagation
        for gconv in self.graph_layers:
            x, _ = gconv(x, edge_index)  # keeps [B, N, hidden_dims[-1]] at the end

        # Readout
        x_pooled = self.global_pooling(x)  # [B, 2*hidden_dims[-1]]

        # Classification
        out = self.classifier(x_pooled)  # [B, C]
        return self.output_act(out)
