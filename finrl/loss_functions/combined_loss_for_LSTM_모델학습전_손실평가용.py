import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

# ✅ Warmup Cosine Scheduler
class WarmupCosineAnnealingScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            warmup_factor = (step + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

# ✅ Dataset
class DualBranchDataset(Dataset):
    def __init__(self, data_dir):
        self.x_minute = np.load(os.path.join(data_dir, "SOLUSDT_1m_finrl_20250101~20250426_X_minute_full.npy"))
        self.x_daily = np.load(os.path.join(data_dir, "SOLUSDT_1m_finrl_20250101~20250426_X_daily_full.npy"))
        self.y_entry = np.load(os.path.join(data_dir, "SOLUSDT_1m_finrl_20250101~20250426_y_entry_full.npy"))
        self.y_direction = np.load(os.path.join(data_dir, "SOLUSDT_1m_finrl_20250101~20250426_y_direction_full.npy"))

    def __len__(self):
        return len(self.y_entry)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_minute[idx], dtype=torch.float32),
            torch.tensor(self.x_daily[idx], dtype=torch.float32),
            torch.tensor(self.y_entry[idx], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.y_direction[idx], dtype=torch.long)
        )

# ✅ 모델
class DualBranchBidirectionalAttentionLSTM_v2(nn.Module):
    def __init__(self, input_size_minute, input_size_daily, hidden_size=128, num_heads=4, dropout_prob=0.3):
        super().__init__()
        self.lstm_minute = nn.LSTM(input_size_minute, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_daily = nn.LSTM(input_size_daily, hidden_size, batch_first=True, bidirectional=True)
        self.attention_minute = nn.MultiheadAttention(hidden_size*2, num_heads, batch_first=True)
        self.attention_daily = nn.MultiheadAttention(hidden_size*2, num_heads, batch_first=True)

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
        out_minute, _ = self.lstm_minute(x_minute)
        out_minute, _ = self.attention_minute(out_minute, out_minute, out_minute)
        out_minute_pooled = out_minute.mean(dim=1)

        out_daily, _ = self.lstm_daily(x_daily)
        out_daily, _ = self.attention_daily(out_daily, out_daily, out_daily)
        out_daily_pooled = out_daily.mean(dim=1)

        combined = torch.cat([out_minute_pooled, out_daily_pooled], dim=1)
        common_feat = self.fc_common(combined)

        return self.fc_entry(common_feat), self.fc_direction(common_feat)

# ✅ 손실 함수
class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        prob = torch.sigmoid(inputs)
        focal = (1 - prob) ** self.gamma * bce
        return focal.mean()

class SoftmaxFocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        targets = F.one_hot(targets, num_classes).float()
        targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        focal = -targets * ((1 - probs) ** self.gamma) * log_probs
        return focal.sum(dim=1).mean()

class CombinedLoss(nn.Module):
    def __init__(self, entry_loss_fn, direction_loss_fn, alpha=1.0, beta=2.0):
        super().__init__()
        self.entry_loss_fn = entry_loss_fn
        self.direction_loss_fn = direction_loss_fn
        self.alpha = alpha
        self.beta = beta

    def forward(self, entry_pred, entry_label, direction_pred, direction_label):
        loss_entry = self.entry_loss_fn(entry_pred, entry_label)
        loss_direction = self.direction_loss_fn(direction_pred, direction_label)
        return self.alpha * loss_entry + self.beta * loss_direction, loss_entry, loss_direction

# ✅ Main
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "data/dualbranch"
    dataset = DualBranchDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = DualBranchBidirectionalAttentionLSTM_v2(20, 20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineAnnealingScheduler(optimizer, warmup_steps=10, total_steps=100)
    scaler = GradScaler()

    entry_loss_fn = BinaryFocalLossWithLogits(gamma=2.0)
    direction_loss_fn = SoftmaxFocalLossWithLabelSmoothing(gamma=2.0, label_smoothing=0.1)
    combined_loss = CombinedLoss(entry_loss_fn, direction_loss_fn, alpha=1.0, beta=2.0)

    os.makedirs("testlog", exist_ok=True)
    log_path = f"testlog/combined_loss_with_scheduler_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(log_path, "w") as f:
        for batch in dataloader:
            x_minute, x_daily, y_entry, y_direction = [x.to(device) for x in batch]

            optimizer.zero_grad()
            with autocast():
                entry_pred, direction_pred = model(x_minute, x_daily)
                loss, entry_loss, direction_loss = combined_loss(entry_pred, y_entry, direction_pred, y_direction)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            msg = f"Total: {loss.item():.4f}, Entry: {entry_loss.item():.4f}, Direction: {direction_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
            print(msg)
            f.write(msg + "\n")
