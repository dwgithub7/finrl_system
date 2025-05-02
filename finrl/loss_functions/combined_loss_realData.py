import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from finrl.loss_functions.focal_loss_realData import BinaryFocalLossWithLogits, SoftmaxFocalLossWithLabelSmoothing

from datetime import datetime

from finrl.loss_functions.focal_loss_realData import BinaryFocalLossWithLogits, SoftmaxFocalLossWithLabelSmoothing
from finrl.models.dualbranch_lstm_v2 import DualBranchBidirectionalAttentionLSTM_v2

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

class CombinedLoss(nn.Module):
    def __init__(self, entry_loss_fn, direction_loss_fn, alpha=1.0, beta=2.0):
        super().__init__()
        self.entry_loss_fn = entry_loss_fn
        self.direction_loss_fn = direction_loss_fn
        self.alpha = alpha
        self.beta = beta

    def forward(self, entry_preds, entry_labels, direction_preds, direction_labels):
        entry_loss = self.entry_loss_fn(entry_preds, entry_labels)
        direction_loss = self.direction_loss_fn(direction_preds, direction_labels)
        total_loss = self.alpha * entry_loss + self.beta * direction_loss
        return total_loss, entry_loss, direction_loss

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "data/dualbranch/"
    dataset = DualBranchDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DualBranchBidirectionalAttentionLSTM_v2(input_size_minute=17, input_size_daily=20).to(device)

    entry_criterion = BinaryFocalLossWithLogits(gamma=2.0)
    direction_criterion = SoftmaxFocalLossWithLabelSmoothing(gamma=2.0, label_smoothing=0.1)
    combined_criterion = CombinedLoss(entry_criterion, direction_criterion, alpha=1.0, beta=2.0)

    os.makedirs("testlog", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"testlog/combine_loss_log_{timestamp}.txt"

    with open(log_path, "w") as f:
        for x_minute, x_daily, y_entry, y_direction in dataloader:
            x_minute = x_minute.to(device)
            x_daily = x_daily.to(device)
            y_entry = y_entry.to(device)
            y_direction = y_direction.to(device)

            entry_preds, direction_preds = model(x_minute, x_daily)

            total_loss, entry_loss, direction_loss = combined_criterion(entry_preds, y_entry, direction_preds, y_direction)

            log_text = f"Total Loss: {total_loss.item():.4f}, Entry Loss: {entry_loss.item():.4f}, Direction Loss: {direction_loss.item():.4f}\n"
            print(log_text.strip())
            f.write(log_text)
