
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from lstm_multitask_model_strong import LSTMMultiTaskModel
from tqdm import tqdm

# ✅ 설정값
DATA_DIR = "data/multitask/"
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.003
HIDDEN_SIZE = 128
ALPHA = 2.0  # entry 중요도 ↑
BETA = 1.0   # direction 상대적으로 ↓

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 디바이스:", device)

# 📦 데이터 로딩
X = np.load(DATA_DIR + "SOLUSDT_X.npy")
y_entry = np.load(DATA_DIR + "SOLUSDT_y_entry.npy")
y_direction = np.load(DATA_DIR + "SOLUSDT_y_direction.npy")

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
    total_entry_loss = 0
    total_dir_loss = 0
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
            loss_dir = torch.tensor(0.0, device=device)

        loss = ALPHA * loss_entry + BETA * loss_dir

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_entry_loss += loss_entry.item()
        total_dir_loss += loss_dir.item()

    print(f"📘 Epoch {epoch+1} | Total Loss: {total_loss:.4f} | Entry: {total_entry_loss:.4f} | Dir: {total_dir_loss:.4f}")
