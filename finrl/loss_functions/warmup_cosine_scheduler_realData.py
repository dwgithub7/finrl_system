import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from datetime import datetime

# ✅ Warmup + CosineAnnealing Scheduler
class WarmupCosineAnnealingScheduler(_LRScheduler):
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

# ✅ Dataset 클래스
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
        y_entry = torch.tensor(self.y_entry[idx], dtype=torch.float32).unsqueeze(0)
        y_direction = torch.tensor(self.y_direction[idx], dtype=torch.long)
        return x_minute, x_daily, y_entry, y_direction

# ✅ 간단 모델 (테스트용)
class SimpleModel(nn.Module):
    def __init__(self, input_size_minute, input_size_daily):
        super().__init__()
        self.fc = nn.Linear(input_size_minute + input_size_daily, 2)

    def forward(self, x_minute, x_daily):
        x_minute_pooled = x_minute.mean(dim=1)
        x_daily_pooled = x_daily.mean(dim=1)
        x = torch.cat([x_minute_pooled, x_daily_pooled], dim=1)
        return self.fc(x)

# ✅ Main
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "data/dualbranch/"
    dataset = DualBranchDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleModel(input_size_minute=17, input_size_daily=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineAnnealingScheduler(optimizer, warmup_steps=10, total_steps=100, min_lr=1e-6)
    scaler = GradScaler()

    # 출력 디렉토리 준비
    os.makedirs("testlog", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"testlog/warmup_cosine_log_{timestamp}.txt"

    with open(log_path, "w") as f:
        for epoch in range(1):
            model.train()
            for x_minute, x_daily, y_entry, y_direction in dataloader:
                x_minute = x_minute.to(device)
                x_daily = x_daily.to(device)
                y_entry = y_entry.to(device)
                y_direction = y_direction.to(device)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(x_minute, x_daily)
                    loss = nn.CrossEntropyLoss()(outputs, y_direction)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                current_lr = scheduler.get_last_lr()[0]
                log_text = f"Loss: {loss.item():.4f}, LR: {current_lr:.8f}\n"
                print(log_text.strip())
                f.write(log_text)
