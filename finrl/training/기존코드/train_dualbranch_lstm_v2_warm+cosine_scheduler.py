import torch
import torch.nn as nn
import torch.optim as optim

from finrl.models.dualbranch_lstm_v2_usedDummyData import DualBranchBidirectionalAttentionLSTM_v2
from finrl.loss_functions.combined_loss import CombinedLoss
from finrl.loss_functions.focal_loss_dummyData import BinaryFocalLossWithLogits, SoftmaxFocalLossWithLabelSmoothing
from finrl.loss_functions.warmup_cosine_scheduler import WarmupCosineAnnealingScheduler

# ✅ Trainer Function

def train_one_epoch(model, dataloader, optimizer, scheduler, combined_loss_fn, device="cuda"):
    model.train()

    total_loss = 0
    total_entry_loss = 0
    total_direction_loss = 0

    for batch in dataloader:
        x_minute, x_daily, entry_labels, direction_labels = batch
        x_minute = x_minute.to(device)
        x_daily = x_daily.to(device)
        entry_labels = entry_labels.to(device)
        direction_labels = direction_labels.to(device)

        optimizer.zero_grad()

        entry_preds, direction_preds = model(x_minute, x_daily)

        loss, entry_loss, direction_loss = combined_loss_fn(entry_preds, entry_labels, direction_preds, direction_labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_entry_loss += entry_loss.item()
        total_direction_loss += direction_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_entry_loss = total_entry_loss / len(dataloader)
    avg_direction_loss = total_direction_loss / len(dataloader)

    return avg_loss, avg_entry_loss, avg_direction_loss

# ✅ Usage Example (Dummy DataLoader)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DualBranchBidirectionalAttentionLSTM_v2(input_size_minute=17, input_size_daily=20).to(device)

    entry_criterion = BinaryFocalLossWithLogits(gamma=2.0)
    direction_criterion = SoftmaxFocalLossWithLabelSmoothing(gamma=2.0, label_smoothing=0.1)
    combined_criterion = CombinedLoss(entry_criterion, direction_criterion, alpha=1.0, beta=2.0)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineAnnealingScheduler(optimizer, warmup_steps=10, total_steps=100, min_lr=1e-6)

    # Dummy Dataloader
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            x_minute = torch.randn(30, 17)
            x_daily = torch.randn(10, 20)
            entry_label = torch.randint(0, 2, (1,)).float()
            direction_label = torch.randint(0, 2, (1,)).squeeze(0)
            return x_minute, x_daily, entry_label, direction_label

    dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=32, shuffle=True)

    # Train one epoch
    avg_loss, avg_entry_loss, avg_direction_loss = train_one_epoch(model, dataloader, optimizer, scheduler, combined_criterion, device)

    print(f"Average Total Loss: {avg_loss:.4f}")
    print(f"Average Entry Loss: {avg_entry_loss:.4f}")
    print(f"Average Direction Loss: {avg_direction_loss:.4f}")
