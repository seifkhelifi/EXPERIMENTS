import torch
import torch.nn as nn


class CNN_GRU(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        feature_layers=64,
        gru_units=128,
        dropout_rate=0.4,
    ):
        super(CNN_GRU, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, feature_layers, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(feature_layers, feature_layers // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        self.gru = nn.GRU(
            input_size=feature_layers // 2,
            hidden_size=gru_units,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(gru_units, num_classes)
        self.output_act = nn.Sigmoid() if num_classes == 1 else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]

        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]  # Last timestep

        x = self.fc(gru_out)
        return self.output_act(x)
