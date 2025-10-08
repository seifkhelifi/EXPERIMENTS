import torch
import torch.nn as nn


class BiLSTM_CNN(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        feature_layers=64,
        lstm_units=128,
        dropout_rate=0.4,
    ):
        super(BiLSTM_CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, feature_layers, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_layers),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(feature_layers, feature_layers // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        # BiLSTM Configuration
        self.bilstm = nn.LSTM(
            input_size=feature_layers // 2,
            hidden_size=lstm_units,
            num_layers=2,  # Deeper LSTM
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if 2 > 1 else 0,
        )

        # Bidirectional output is 2*lstm_units
        self.fc = nn.Linear(lstm_units * 2, num_classes)
        self.output_act = nn.Sigmoid() if num_classes == 1 else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]

        bilstm_out, _ = self.bilstm(x)
        # Use mean pooling over time instead of last step
        bilstm_out = torch.mean(bilstm_out, dim=1)

        x = self.fc(bilstm_out)
        return self.output_act(x)
