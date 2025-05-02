import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# ✅ 모델 클래스 (변경 없음)
class DualBranchBidirectionalAttentionLSTM_v2(nn.Module):
    def __init__(self, input_size_minute, input_size_daily, hidden_size=128, num_heads=4, dropout_prob=0.3):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm_minute = nn.LSTM(input_size_minute, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_daily = nn.LSTM(input_size_daily, hidden_size, batch_first=True, bidirectional=True)

        self.attention_minute = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=num_heads, batch_first=True)
        self.attention_daily = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=num_heads, batch_first=True)

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

        self.fc_entry = nn.Linear(64, 1)
        self.fc_direction = nn.Linear(64, 2)

    def forward(self, x_minute, x_daily):
        assert x_minute.ndim == 3 and x_daily.ndim == 3, "Inputs must be 3D tensors."

        out_minute, _ = self.lstm_minute(x_minute)
        out_minute, _ = self.attention_minute(out_minute, out_minute, out_minute)
        out_minute_pooled = out_minute.mean(dim=1)

        out_daily, _ = self.lstm_daily(x_daily)
        out_daily, _ = self.attention_daily(out_daily, out_daily, out_daily)
        out_daily_pooled = out_daily.mean(dim=1)

        combined = torch.cat([out_minute_pooled, out_daily_pooled], dim=1)
        common_feat = self.fc_common(combined)

        entry_logit = self.fc_entry(common_feat)
        direction_logit = self.fc_direction(common_feat)

        entry_prob = torch.sigmoid(entry_logit)
        direction_prob = torch.softmax(direction_logit, dim=1)

        return entry_prob, direction_prob

# ✅ Dataset 클래스 (새로 추가)
class DualBranchDataset(Dataset):
    def __init__(self, data_dir):
        self.x_minute = np.load(os.path.join(data_dir, "SOLUSDT_1m_finrl_20250101~20250426_X_minute_full.npy"))
        self.x_daily = np.load(os.path.join(data_dir, "SOLUSDT_1m_finrl_20250101~20250426_X_daily_full.npy"))
        self.y_entry = np.load(os.path.join(data_dir, "SOLUSDT_1m_finrl_20250101~20250426_y_entry_full.npy"))
        self.y_direction = np.load(os.path.join(data_dir, "SOLUSDT_1m_finrl_20250101~20250426_y_direction_full.npy"))

    def __len__(self):
        return len(self.y_entry)

    def __getitem__(self, idx):
        x_minute = torch.tensor(self.x_minute[idx], dtype=torch.float32)
        x_daily = torch.tensor(self.x_daily[idx], dtype=torch.float32)
        y_entry = torch.tensor(self.y_entry[idx], dtype=torch.float32).unsqueeze(0)  # (1,) 형태
        y_direction = torch.tensor(self.y_direction[idx], dtype=torch.long)
        return x_minute, x_daily, y_entry, y_direction

# ✅ Usage Example (실전 데이터 기반)
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "data/dualbranch/"
    dataset = DualBranchDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DualBranchBidirectionalAttentionLSTM_v2(input_size_minute=17, input_size_daily=20).to(device)

    os.makedirs("testlog", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"testlog/dualbranch_lstm_v2_log_{timestamp}.txt"

    with open(log_path, "w") as f:
        for x_minute, x_daily, y_entry, y_direction in dataloader:
            x_minute = x_minute.to(device)
            x_daily = x_daily.to(device)
            y_entry = y_entry.to(device)
            y_direction = y_direction.to(device)

            entry_prob, direction_prob = model(x_minute, x_daily)

            log_text = f"Entry output shape: {entry_prob.shape}, Direction output shape: {direction_prob.shape}\n"
            print(log_text.strip())
            f.write(log_text)
