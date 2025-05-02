
import torch
import torch.nn as nn

class LSTMMultiTaskModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        # 진입 판단: binary classification
        self.head_entry = nn.Linear(hidden_size, 1)

        # 방향 예측: -1 or +1 → class 0 or 1 (softmax)
        self.head_direction = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        last_output = out[:, -1, :]  # 마지막 타임스텝만 사용
        last_output = self.dropout(last_output)

        entry_logit = self.head_entry(last_output)              # (batch, 1)
        direction_logits = self.head_direction(last_output)     # (batch, 2)

        entry_prob = torch.sigmoid(entry_logit)                 # (batch, 1)
        direction_prob = torch.softmax(direction_logits, dim=1) # (batch, 2)

        return entry_prob, direction_prob
