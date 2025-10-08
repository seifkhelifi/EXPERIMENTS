import torch
import torch.nn as nn


class TransformerIDS(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        feature_dim=128,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.3,
        ff_hidden=256,
    ):
        super(TransformerIDS, self).__init__()

        self.num_classes = num_classes

        # CNN encoder: local feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 65 // 2, feature_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

        # Output activation depending on task
        self.output_act = nn.Sigmoid() if num_classes == 1 else nn.Identity()

    def forward(self, x):
        # Input: [batch, seq_len, features] → [batch, features, seq_len]
        x = x.permute(0, 2, 1)

        # CNN
        x = self.cnn(x)  # [batch, feature_dim, seq_len']
        x = x.permute(0, 2, 1)  # [batch, seq_len', feature_dim]

        # Add positional encoding
        x = x + self.pos_embedding[:, : x.size(1), :]

        # Transformer encoder
        x = self.transformer(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Classification
        x = self.fc(x)
        return self.output_act(x)
