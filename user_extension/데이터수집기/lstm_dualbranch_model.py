
import torch
import torch.nn as nn

class DualBranchLSTM(nn.Module):
    def __init__(self, input_size_minute, input_size_daily, hidden_size=128):
        super(DualBranchLSTM, self).__init__()
        self.lstm_minute = nn.LSTM(input_size_minute, hidden_size, batch_first=True)
        self.lstm_daily = nn.LSTM(input_size_daily, hidden_size, batch_first=True)

        self.fc_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.fc_entry = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.fc_direction = nn.Sequential(
            nn.Linear(hidden_size, 2)  # binary classification
        )

    def forward(self, x_minute, x_daily):
        _, (h_minute, _) = self.lstm_minute(x_minute)
        _, (h_daily, _) = self.lstm_daily(x_daily)

        h_combined = torch.cat([h_minute[-1], h_daily[-1]], dim=1)
        fusion = self.fc_fusion(h_combined)

        entry = self.fc_entry(fusion)
        direction = self.fc_direction(fusion)
        return entry, direction
