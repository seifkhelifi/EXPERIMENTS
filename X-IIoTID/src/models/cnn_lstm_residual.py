import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Residual connection
        out = self.relu(out)
        return self.dropout(out)


class DeepResidual_CNN_LSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        feature_layers=64,
        lstm_units=128,
        dropout_rate=0.4,
    ):
        super(DeepResidual_CNN_LSTM, self).__init__()

        # Initial conv layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, feature_layers, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_layers),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        # Residual blocks
        self.res_block1 = ResidualBlock(feature_layers, feature_layers * 2)
        self.res_block2 = ResidualBlock(feature_layers * 2, feature_layers * 4)

        # Calculate LSTM input size
        dummy = torch.zeros(1, input_channels, 65)
        cnn_out = self.res_block2(self.res_block1(self.conv1(dummy)))
        seq_len = cnn_out.shape[2]

        # LSTM
        self.lstm = nn.LSTM(
            input_size=feature_layers * 4, hidden_size=lstm_units, batch_first=True
        )

        self.fc = nn.Linear(lstm_units, num_classes)
        self.output_act = nn.Sigmoid() if num_classes == 1 else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Last timestep
        x = self.fc(lstm_out)
        return self.output_act(x)
