
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from lstm_dualbranch_model import DualBranchLSTM
from sklearn.utils import class_weight

# ✅ 설정
DATA_DIR = "data/multitask/"
EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 0.003
HIDDEN_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 디바이스:", device)

# ✅ 데이터 로딩
X_minute = np.load(DATA_DIR + "X_minute.npy")[:, :, :20]
X_daily = np.load(DATA_DIR + "X_daily.npy")
y_entry = np.load(DATA_DIR + "y_entry.npy")
y_direction = np.load(DATA_DIR + "y_direction.npy")

# ✅ entry == 1인 샘플만 추출
mask = y_entry == 1
X_minute = X_minute[mask]
X_daily = X_daily[mask]
y_direction = y_direction[mask]

# ✅ 텐서로 변환
X_minute_tensor = torch.tensor(X_minute, dtype=torch.float32)
X_daily_tensor = torch.tensor(X_daily, dtype=torch.float32)
y_direction_tensor = torch.tensor(y_direction, dtype=torch.long)

# ✅ 데이터셋/로더 정의
dataset = TensorDataset(X_minute_tensor, X_daily_tensor, y_direction_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ 클래스 가중치 계산
class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_direction), y=y_direction)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# ✅ 모델 구성
model = DualBranchLSTM(input_size_minute=20, input_size_daily=23, hidden_size=HIDDEN_SIZE)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# ✅ 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x_min, x_day, y_dir in loader:
        x_min = x_min.to(device)
        x_day = x_day.to(device)
        y_dir = y_dir.to(device)

        _, dir_pred = model(x_min, x_day)
        loss = loss_fn(dir_pred, y_dir)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"📘 Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# ✅ 저장
torch.save(model.state_dict(), "saved_direction_only_model.pt")
print("✅ direction-only 모델 저장 완료!")
