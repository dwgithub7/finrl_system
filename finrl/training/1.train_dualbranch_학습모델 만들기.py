import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# ✅ 경로 및 기본 파일 이름 설정
data_dir = "data/dualbranch"
base_filename = "SOLUSDT_1m_finrl_20240101~20250426"  # 🔥 여기를 상황에 맞게 바꿔주기
model_save_path = "data/models/dualbranch_model.pth"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# ✅ 데이터 로드
X_minute = np.load(os.path.join(data_dir, f"{base_filename}_X_minute.npy"))
X_daily = np.load(os.path.join(data_dir, f"{base_filename}_X_daily.npy"))
y_entry = np.load(os.path.join(data_dir, f"{base_filename}_y_entry.npy"))
y_direction = np.load(os.path.join(data_dir, f"{base_filename}_y_direction.npy"))

# ✅ Tensor 변환
X_minute = torch.tensor(X_minute, dtype=torch.float32)
X_daily = torch.tensor(X_daily, dtype=torch.float32)
y_entry = torch.tensor(y_entry, dtype=torch.float32).unsqueeze(1)
y_direction = torch.tensor(y_direction, dtype=torch.long)

# ✅ Dataset 생성
dataset = TensorDataset(X_minute, X_daily, y_entry, y_direction)

# ✅ train/val 분할
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ✅ DataLoader 생성
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ✅ DualBranch 모델 정의
class DualBranchLSTM(nn.Module):
    def __init__(self, input_size_minute, input_size_daily, hidden_size=128):
        super().__init__()
        self.lstm_minute = nn.LSTM(input_size_minute, hidden_size, batch_first=True)
        self.lstm_daily = nn.LSTM(input_size_daily, hidden_size, batch_first=True)
        self.fc_entry = nn.Linear(hidden_size * 2, 1)
        self.fc_direction = nn.Linear(hidden_size * 2, 2)

    def forward(self, x_minute, x_daily):
        _, (h_minute, _) = self.lstm_minute(x_minute)
        _, (h_daily, _) = self.lstm_daily(x_daily)
        h = torch.cat([h_minute[-1], h_daily[-1]], dim=1)
        entry_out = torch.sigmoid(self.fc_entry(h))
        dir_out = self.fc_direction(h)
        return entry_out, dir_out

# ✅ 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ✅ 모델 생성
input_size_minute = X_minute.shape[2]
input_size_daily = X_daily.shape[2]
model = DualBranchLSTM(input_size_minute, input_size_daily).to(device)

# ✅ Loss 함수 및 Optimizer
loss_entry_fn = nn.BCELoss()
loss_direction_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ✅ 학습 루프
num_epochs = 30
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for x_min, x_day, y_e, y_d in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
        x_min, x_day, y_e, y_d = x_min.to(device), x_day.to(device), y_e.to(device), y_d.to(device)

        pred_entry, pred_dir = model(x_min, x_day)

        loss_entry = loss_entry_fn(pred_entry, y_e)
        mask = (y_e.squeeze() > 0.5)
        if mask.sum() > 0:
            loss_dir = loss_direction_fn(pred_dir[mask], y_d[mask])
        else:
            loss_dir = torch.tensor(0.0, device=device)

        loss = loss_entry + loss_dir

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ✅ 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_min, x_day, y_e, y_d in val_loader:
            x_min, x_day, y_e, y_d = x_min.to(device), x_day.to(device), y_e.to(device), y_d.to(device)

            pred_entry, pred_dir = model(x_min, x_day)
            loss_entry = loss_entry_fn(pred_entry, y_e)
            mask = (y_e.squeeze() > 0.5)
            if mask.sum() > 0:
                loss_dir = loss_direction_fn(pred_dir[mask], y_d[mask])
            else:
                loss_dir = torch.tensor(0.0, device=device)

            loss = loss_entry + loss_dir
            val_loss += loss.item()

    print(f"\n📘 Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    # ✅ Best 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Best model saved at epoch {epoch+1}")

print("🎯 학습 완료.")

