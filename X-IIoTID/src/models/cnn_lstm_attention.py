import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden_dim]
        energy = torch.tanh(self.attn(lstm_out))  # [batch, seq_len, hidden_dim]
        energy = energy.permute(0, 2, 1)  # [batch, hidden_dim, seq_len]
        v = self.v.repeat(lstm_out.size(0), 1).unsqueeze(1)  # [batch, 1, hidden_dim]
        attention = torch.bmm(v, energy).squeeze(1)  # [batch, seq_len]
        return torch.softmax(attention, dim=1)


class CNN_LSTM_Attention(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        feature_layers=64,
        lstm_units=128,
        dropout_rate=0.4,
    ):
        super(CNN_LSTM_Attention, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, feature_layers, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        self.lstm = nn.LSTM(
            input_size=feature_layers, hidden_size=lstm_units, batch_first=True
        )

        self.attention = TemporalAttention(lstm_units)
        self.fc = nn.Linear(lstm_units, num_classes)
        self.output_act = nn.Sigmoid() if num_classes == 1 else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]

        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_units]

        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # [batch, seq_len]
        attn_weights = attn_weights.unsqueeze(2)  # [batch, seq_len, 1]
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, lstm_units]

        x = self.fc(context)
        return self.output_act(x)
