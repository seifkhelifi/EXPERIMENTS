import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        feature_layers=64,
        lstm_units=128,
        dropout_rate=0.4,
        kernel_size=3,
        pool_size=2,
    ):
        super(CNN_LSTM, self).__init__()

        self.num_classes = num_classes

        # CNN Part
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=feature_layers,
                kernel_size=kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate),
        )

        # Calculate output size after Conv1d + MaxPool1d
        dummy_input = torch.zeros(
            1, input_channels, 65
        )  # (batch, channels, sequence length)
        cnn_output = self.cnn(dummy_input)
        self.sequence_len = cnn_output.shape[2]

        # LSTM
        self.lstm = nn.LSTM(
            input_size=feature_layers, hidden_size=lstm_units, batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(lstm_units, num_classes)
        if self.num_classes == 1:
            self.output_act = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, 1) → (batch, 1, seq_len)
        x = x.permute(0, 2, 1)

        # CNN: (batch, 1, 65) → (batch, 64, seq_len_out)
        x = self.cnn(x)

        # Prepare for LSTM: (batch, 64, seq_len_out) → (batch, seq_len_out, 64)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step

        x = self.fc(lstm_out)
        return self.output_act(x) if self.num_classes == 1 else x
