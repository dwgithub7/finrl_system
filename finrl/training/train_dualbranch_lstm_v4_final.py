import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
from datetime import datetime

from finrl.models.dualbranch_lstm_v2 import DualBranchBidirectionalAttentionLSTM_v2
from finrl.loss_functions.combined_loss_realData import CombinedLoss
from finrl.loss_functions.focal_loss_dummyData import BinaryFocalLossWithLogits, SoftmaxFocalLossWithLabelSmoothing

# ✅ Dataset
class RealDualBranchDataset(Dataset):
    def __init__(self, data_dir):
        self.x_minute = np.load(os.path.join(data_dir, 'SOLUSDT_1m_finrl_20250101~20250426_X_minute_full.npy'))
        self.x_daily = np.load(os.path.join(data_dir, 'SOLUSDT_1m_finrl_20250101~20250426_X_daily_full.npy'))
        self.y_entry = np.load(os.path.join(data_dir, 'SOLUSDT_1m_finrl_20250101~20250426_y_entry_full.npy'))
        self.y_direction = np.load(os.path.join(data_dir, 'SOLUSDT_1m_finrl_20250101~20250426_y_direction_full.npy'))

        assert not np.isnan(self.x_minute).any(), "x_minute contains NaNs!"
        assert not np.isnan(self.x_daily).any(), "x_daily contains NaNs!"
        assert not np.isnan(self.y_entry).any(), "y_entry contains NaNs!"
        assert not np.isnan(self.y_direction).any(), "y_direction contains NaNs!"

    def __len__(self):
        return len(self.y_entry)

    def __getitem__(self, idx):
        x_minute = torch.tensor(self.x_minute[idx], dtype=torch.float32)
        x_daily = torch.tensor(self.x_daily[idx], dtype=torch.float32)
        entry_label = torch.tensor(self.y_entry[idx], dtype=torch.float32)
        direction_label = torch.tensor(self.y_direction[idx], dtype=torch.long)
        return x_minute, x_daily, entry_label, direction_label

# ✅ Warmup Cosine Annealing Scheduler
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

# ✅ Trainer
class DualBranchLSTMTrainer:
    def __init__(self, input_size_minute, input_size_daily, model_dir="models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.model = DualBranchBidirectionalAttentionLSTM_v2(input_size_minute, input_size_daily).to(self.device)

    def train_one_epoch(self, dataloader, optimizer, scheduler, combined_loss_fn, scaler, max_grad_norm=1.0):
        self.model.train()

        total_loss = 0
        total_entry_loss = 0
        total_direction_loss = 0

        for batch in dataloader:
            x_minute, x_daily, entry_labels, direction_labels = batch
            x_minute = x_minute.to(self.device)
            x_daily = x_daily.to(self.device)
            entry_labels = entry_labels.to(self.device)
            direction_labels = direction_labels.to(self.device)

            optimizer.zero_grad()

            with autocast():
                entry_preds, direction_preds = self.model(x_minute, x_daily)
                loss, entry_loss, direction_loss = combined_loss_fn(entry_preds, entry_labels, direction_preds, direction_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_entry_loss += entry_loss.item()
            total_direction_loss += direction_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_entry_loss = total_entry_loss / len(dataloader)
        avg_direction_loss = total_direction_loss / len(dataloader)

        return avg_loss, avg_entry_loss, avg_direction_loss

    def save(self, model_name="dualbranch_lstm"):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, f"{model_name}_model.pth"))

    def load(self, model_name="dualbranch_lstm"):
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, f"{model_name}_model.pth"), map_location=self.device))

    def predict(self, X_minute_last, X_daily_last):
        self.model.eval()
        with torch.no_grad():
            x_minute = torch.tensor(X_minute_last, dtype=torch.float32).to(self.device)
            x_daily = torch.tensor(X_daily_last, dtype=torch.float32).to(self.device)
            entry_logits, direction_logits = self.model(x_minute, x_daily)
            entry_probs = torch.sigmoid(entry_logits).cpu().numpy().flatten()
            direction_probs = torch.sigmoid(direction_logits).cpu().numpy().flatten()
        return entry_probs, direction_probs


# ✅ Main 실행
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = DualBranchLSTMTrainer(input_size_minute=17, input_size_daily=20)
    model = trainer.model

    entry_criterion = BinaryFocalLossWithLogits(gamma=2.0)
    direction_criterion = SoftmaxFocalLossWithLabelSmoothing(gamma=2.0, label_smoothing=0.1)
    combined_criterion = CombinedLoss(entry_criterion, direction_criterion, alpha=1.0, beta=2.0)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineAnnealingScheduler(optimizer, warmup_steps=10, total_steps=100, min_lr=1e-6)
    scaler = GradScaler()

    dataset = RealDualBranchDataset("data/dualbranch")
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

    avg_loss, avg_entry_loss, avg_direction_loss = trainer.train_one_epoch(
        dataloader, optimizer, scheduler, combined_criterion, scaler
    )

    print(f"Average Total Loss: {avg_loss:.4f}")
    print(f"Average Entry Loss: {avg_entry_loss:.4f}")
    print(f"Average Direction Loss: {avg_direction_loss:.4f}")

    trainer.save(model_name="dualbranch_lstm")
    trainer.load(model_name="dualbranch_lstm")

    print(" Model saved and reloaded successfully.")
