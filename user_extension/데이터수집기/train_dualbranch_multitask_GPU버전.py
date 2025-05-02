
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from lstm_dualbranch_model import DualBranchLSTM
from tqdm import tqdm

# ✅ 설정값
DATA_DIR = "data/multitask/"
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.003
HIDDEN_SIZE = 128
ALPHA = 1.0
BETA = 2.0  # direction 손실 가중치 ↑

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 디바이스:", device)

# 📦 데이터 로딩 (1분봉 + 일봉)
X_minute = np.load(DATA_DIR + "X_minute.npy")  # (N, 60, F1)
X_daily = np.load(DATA_DIR + "X_daily.npy")    # (N, 20, F2)
y_entry = np.load(DATA_DIR + "y_entry.npy")
y_direction = np.load(DATA_DIR + "y_direction.npy")

X_minute_tensor = torch.tensor(X_minute, dtype=torch.float32)
X_daily_tensor = torch.tensor(X_daily, dtype=torch.float32)
y_entry_tensor = torch.tensor(y_entry, dtype=torch.float32).unsqueeze(1)
y_direction_tensor = torch.tensor((y_direction == 1).astype(int), dtype=torch.long)

dataset = TensorDataset(X_minute_tensor, X_daily_tensor, y_entry_tensor, y_direction_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ 모델 정의
input_size_minute = X_minute.shape[2]
input_size_daily = X_daily.shape[2]
model = DualBranchLSTM(input_size_minute=input_size_minute, input_size_daily=input_size_daily, hidden_size=HIDDEN_SIZE)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_entry_fn = nn.BCELoss()
loss_direction_fn = nn.CrossEntropyLoss()

entry_losses_per_epoch = []
dir_losses_per_epoch = []

# ✅ 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_entry_loss = 0
    total_dir_loss = 0
    for x_minute, x_daily, y_entry_batch, y_dir_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x_minute = x_minute.to(device)
        x_daily = x_daily.to(device)
        y_entry_batch = y_entry_batch.to(device)
        y_dir_batch = y_dir_batch.to(device)

        pred_entry, pred_dir = model(x_minute, x_daily)

        loss_entry = loss_entry_fn(pred_entry, y_entry_batch)

        mask = (y_entry_batch.squeeze() > 0.5)
        if mask.sum() > 0:
            loss_dir = loss_direction_fn(pred_dir[mask], y_dir_batch[mask])
        else:
            loss_dir = torch.tensor(0.0, device=device)

        loss = ALPHA * loss_entry + BETA * loss_dir

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_entry_loss += loss_entry.item()
        total_dir_loss += loss_dir.item()

    entry_losses_per_epoch.append(total_entry_loss)
    dir_losses_per_epoch.append(total_dir_loss)
    print(f"📘 Epoch {epoch+1} | Total Loss: {total_loss:.4f} | Entry: {total_entry_loss:.4f} | Dir: {total_dir_loss:.4f}")

# ✅ 여기부터 추가
print("📦 학습 완료, 모델 저장 중...")
torch.save(model.state_dict(), "saved_model.pt")
print("✅ 모델 저장 완료: saved_model.pt")