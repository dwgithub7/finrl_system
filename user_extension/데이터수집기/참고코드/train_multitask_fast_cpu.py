
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from lstm_multitask_model import LSTMMultiTaskModel
from tqdm import tqdm

# ✅ 설정값 (경량 테스트 전용)
DATA_DIR = "data/multitask/"
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
ALPHA = 1.0
BETA = 1.0
MAX_SAMPLES = 3000  # 빠른 테스트용 데이터 수 제한

device = torch.device("cpu")
print("🧠 CPU 전용 테스트 실행 중...")

# 📦 데이터 로딩 (작은 샘플만 사용)
X = np.load(DATA_DIR + "X.npy")[:MAX_SAMPLES]
y_entry = np.load(DATA_DIR + "y_entry.npy")[:MAX_SAMPLES]
y_direction = np.load(DATA_DIR + "y_direction.npy")[:MAX_SAMPLES]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_entry_tensor = torch.tensor(y_entry, dtype=torch.float32).unsqueeze(1)
y_direction_tensor = torch.tensor((y_direction == 1).astype(int), dtype=torch.long)

dataset = TensorDataset(X_tensor, y_entry_tensor, y_direction_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ 모델 정의
input_size = X.shape[2]
model = LSTMMultiTaskModel(input_size=input_size, hidden_size=HIDDEN_SIZE)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_entry_fn = nn.BCELoss()
loss_direction_fn = nn.CrossEntropyLoss()

# ✅ 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_entry_batch, y_dir_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        X_batch = X_batch.to(device)
        y_entry_batch = y_entry_batch.to(device)
        y_dir_batch = y_dir_batch.to(device)

        pred_entry, pred_dir = model(X_batch)

        loss_entry = loss_entry_fn(pred_entry, y_entry_batch)
        mask = (y_entry_batch.squeeze() > 0.5)

        if mask.sum() > 0:
            loss_dir = loss_direction_fn(pred_dir[mask], y_dir_batch[mask])
        else:
            loss_dir = torch.tensor(0.0)

        loss = ALPHA * loss_entry + BETA * loss_dir
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📘 Epoch {epoch+1} | Loss: {total_loss:.4f}")
