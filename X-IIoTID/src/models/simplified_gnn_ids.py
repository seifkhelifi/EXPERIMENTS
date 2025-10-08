import torch
import torch.nn as nn
import torch.nn.functional as F

import math


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


# class SimplifiedGNN_IDS(nn.Module):
#     """
#     Simplified version for easier hyperparameter search
#     Works directly on tabular IDS data without explicit graph construction
#     """

#     def __init__(
#         self,
#         num_classes,
#         input_dim=4,
#         hidden_dim=128,
#         num_graph_layers=2,
#         num_heads=4,
#         dropout_rate=0.3,
#     ):
#         super(SimplifiedGNN_IDS, self).__init__()

#         self.num_classes = num_classes
#         self.num_nodes = 16  # Fixed number of pseudo-nodes

#         # Feature partitioning
#         # self.node_dim = input_dim // self.num_nodes

#         # if input_dim % self.num_nodes != 0:
#         #     self.padding_size = self.num_nodes - (input_dim % self.num_nodes)
#         # else:
#         #     self.padding_size = 0

#         # # Node embedding
#         # self.node_encoder = nn.Linear(
#         #     self.node_dim + (1 if self.padding_size > 0 else 0), hidden_dim
#         # )

#         self.node_dim = math.ceil(input_dim / self.num_nodes)
#         self.padding_size = self.num_nodes * self.node_dim - input_dim  # >= 0
#         self.node_encoder = nn.Linear(self.node_dim, hidden_dim)

#         # Graph layers
#         self.graph_convs = nn.ModuleList(
#             [
#                 AdaptiveGraphLayer(hidden_dim, hidden_dim, num_heads, dropout_rate)
#                 for _ in range(num_graph_layers)
#             ]
#         )

#         # Readout
#         self.readout = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, num_classes),
#         )

#         self.output_act = nn.Sigmoid() if num_classes == 1 else nn.Identity()

#     def forward(self, x, edge_index=None):
#         batch_size = x.shape[0]

#         # Pad if necessary
#         if self.padding_size > 0:
#             x = F.pad(x, (0, self.padding_size))

#         # Reshape to graph nodes
#         x = x.view(batch_size, self.num_nodes, -1)

#         # Encode nodes
#         x_flat = x.view(batch_size * self.num_nodes, -1)
#         x_encoded = self.node_encoder(x_flat)
#         x = x_encoded.view(batch_size, self.num_nodes, -1)

#         # Create fully connected graph
#         row = torch.arange(self.num_nodes, device=x.device).repeat_interleave(
#             self.num_nodes
#         )
#         col = torch.arange(self.num_nodes, device=x.device).repeat(self.num_nodes)
#         edge_index = torch.stack([row, col], dim=0)

#         # Apply graph convolutions
#         for conv in self.graph_convs:
#             x, _ = conv(x, edge_index)

#         # Global pooling (mean + max)
#         mean_pool = torch.mean(x, dim=1)
#         max_pool, _ = torch.max(x, dim=1)
#         x_pooled = torch.cat([mean_pool, max_pool], dim=-1)

#         # Classify
#         out = self.readout(x_pooled)
#         return self.output_act(out)


class SimplifiedGNN_IDS(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim=128,
        num_graph_layers=2,
        num_heads=4,
        dropout_rate=0.3,
        num_nodes=16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # defer creation until we know node_dim from x
        self.node_encoder = None

        self.graph_convs = nn.ModuleList(
            [
                AdaptiveGraphLayer(hidden_dim, hidden_dim, num_heads, dropout_rate)
                for _ in range(num_graph_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        self.output_act = nn.Sigmoid() if num_classes == 1 else nn.Identity()

    def _ensure_node_encoder(self, node_dim, device):
        if self.node_encoder is None:
            self.node_encoder = nn.Linear(node_dim, self.hidden_dim).to(device)
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(self.hidden_dim) for _ in range(len(self.graph_convs))]
            )

    def forward(self, x, edge_index=None):
        B = x.shape[0]

        # If x is flat (B, F), convert to (B, num_nodes, node_dim)
        if x.dim() == 2:
            F_flat = x.shape[1]
            node_dim = math.ceil(F_flat / self.num_nodes)
            pad = self.num_nodes * node_dim - F_flat
            if pad > 0:
                x = F.pad(x, (0, pad))
            x = x.view(B, self.num_nodes, node_dim)
        else:
            node_dim = x.shape[-1]

        device = x.device  # <--- get device from input
        self._ensure_node_encoder(node_dim, device)  # <--- move layer to device

        x = self.node_encoder(x.view(B * self.num_nodes, node_dim)).view(
            B, self.num_nodes, -1
        )

        # fully-connected edges on the SAME device
        row = torch.arange(self.num_nodes, device=device).repeat_interleave(
            self.num_nodes
        )
        col = torch.arange(self.num_nodes, device=device).repeat(self.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        for i, conv in enumerate(self.graph_convs):
            x, _ = conv(x, edge_index)

        mean_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        x_pooled = torch.cat([mean_pool, max_pool], dim=-1)
        out = self.readout(x_pooled)
        return self.output_act(out)
