import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from finrl.models.dualbranch_lstm_v2 import DualBranchBidirectionalAttentionLSTM_v2, DualBranchDataset

# ✅ LSTM Trainer Class
class DualBranchLSTMTrainer:
    def __init__(self, input_size_minute, input_size_daily, model_dir="models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.model = DualBranchBidirectionalAttentionLSTM_v2(input_size_minute, input_size_daily).to(self.device)

    def train(self, dataset_dir, batch_size=64, max_epochs=30):
        dataset = DualBranchDataset(dataset_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        criterion_entry = nn.BCELoss()
        criterion_direction = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        print(f"Device: {self.device}")

        for epoch in tqdm(range(max_epochs), desc="Training LSTM"):
            for x_minute, x_daily, y_entry, y_direction in dataloader:
                x_minute = x_minute.to(self.device)
                x_daily = x_daily.to(self.device)
                y_entry = y_entry.to(self.device)
                y_direction = y_direction.to(self.device)

                optimizer.zero_grad()
                entry_prob, direction_prob = self.model(x_minute, x_daily)

                loss_entry = criterion_entry(entry_prob.squeeze(), y_entry.squeeze())
                loss_direction = criterion_direction(direction_prob, y_direction)
                loss = loss_entry + loss_direction

                loss.backward()
                optimizer.step()

    def predict(self, dataset_dir):
        dataset = DualBranchDataset(dataset_dir)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

        self.model.eval()
        entry_probs = []
        direction_probs = []

        with torch.no_grad():
            for x_minute, x_daily, _, _ in tqdm(dataloader, desc="Predicting LSTM"):
                x_minute = x_minute.to(self.device)
                x_daily = x_daily.to(self.device)

                entry_prob, direction_prob = self.model(x_minute, x_daily)
                entry_probs.append(entry_prob.squeeze().cpu().numpy())
                direction_probs.append(direction_prob[:, 1].cpu().numpy())  # Class 1 확률

        entry_probs = np.concatenate(entry_probs)
        direction_probs = np.concatenate(direction_probs)
        return entry_probs, direction_probs

    def save(self, model_name="dualbranch_lstm_LightGBM_TabNet"):
        os.makedirs(self.model_dir, exist_ok=True)
        save_path = os.path.join(self.model_dir, f"{model_name}_model.pth")
        torch.save(self.model.state_dict(), save_path)

    def load(self, model_name="dualbranch_lstm_LightGBM_TabNet"):
        load_path = os.path.join(self.model_dir, f"{model_name}_model.pth")
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))

# ✅ Usage Example
if __name__ == "__main__":
    DATA_DIR = "data/dualbranch/"

    trainer = DualBranchLSTMTrainer(input_size_minute=17, input_size_daily=20)

    os.makedirs("testlog", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"testlog/train_dualbranch_lstm_log_{timestamp}.txt"

    with open(log_path, "w") as f:
        trainer.train(DATA_DIR, batch_size=64, max_epochs=30)
        entry_probs, direction_probs = trainer.predict(DATA_DIR)

        f.write(f"✅ Entry/Direction prediction completed.\n")

        for i in tqdm(range(min(10, len(entry_probs))), desc="Logging Predictions"):
            log_text = f"Entry Prob: {entry_probs[i]:.4f}, Direction Prob: {direction_probs[i]:.4f}\n"
            print(log_text.strip())
            f.write(log_text)

    trainer.save()
    trainer.load()

    print(" All process completed.")

