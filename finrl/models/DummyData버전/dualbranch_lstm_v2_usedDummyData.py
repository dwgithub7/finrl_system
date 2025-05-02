import torch
import torch.nn as nn

class DualBranchBidirectionalAttentionLSTM_v2(nn.Module):
    def __init__(self, input_size_minute, input_size_daily, hidden_size=128, num_heads=4, dropout_prob=0.3):
        super().__init__()
        self.hidden_size = hidden_size

        # Bi-LSTM Layers
        self.lstm_minute = nn.LSTM(input_size_minute, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_daily = nn.LSTM(input_size_daily, hidden_size, batch_first=True, bidirectional=True)

        # Attention Layers
        self.attention_minute = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=num_heads, batch_first=True)
        self.attention_daily = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=num_heads, batch_first=True)

        # Fully Connected Layers with BatchNorm and Dropout
        self.fc_common = nn.Sequential(
            nn.Linear(hidden_size*4, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

        self.fc_entry = nn.Linear(64, 1)       # Binary output for entry
        self.fc_direction = nn.Linear(64, 2)   # Binary classification (0/1) for direction

    def forward(self, x_minute, x_daily):
        # Check input dimensions
        assert x_minute.ndim == 3 and x_daily.ndim == 3, "Inputs must be 3D tensors."

        # 1m branch
        out_minute, _ = self.lstm_minute(x_minute)
        out_minute, _ = self.attention_minute(out_minute, out_minute, out_minute)
        out_minute_pooled = out_minute.mean(dim=1)  # Mean pooling

        # 1d branch
        out_daily, _ = self.lstm_daily(x_daily)
        out_daily, _ = self.attention_daily(out_daily, out_daily, out_daily)
        out_daily_pooled = out_daily.mean(dim=1)

        # Combine branches
        combined = torch.cat([out_minute_pooled, out_daily_pooled], dim=1)
        common_feat = self.fc_common(combined)

        # Heads
        entry_logit = self.fc_entry(common_feat)
        direction_logit = self.fc_direction(common_feat)

        # Apply activation
        entry_prob = torch.sigmoid(entry_logit)
        direction_prob = torch.softmax(direction_logit, dim=1)

        return entry_prob, direction_prob

# ✅ Usage Example (Dummy Input)
if __name__ == "__main__":
    model = DualBranchBidirectionalAttentionLSTM_v2(input_size_minute=17, input_size_daily=20)
    x_min = torch.randn(32, 30, 17)   # (batch, seq_len, features)
    x_day = torch.randn(32, 10, 20)

    entry_prob, dir_prob = model(x_min, x_day)
    print(entry_prob.shape, dir_prob.shape)  # (32,1), (32,2)
