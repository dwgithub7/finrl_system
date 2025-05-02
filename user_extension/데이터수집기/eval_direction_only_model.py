
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from lstm_dualbranch_model import DualBranchLSTM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ 설정
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 디바이스:", DEVICE)

# ✅ 데이터 로딩
X_minute = np.load("data/multitask/X_minute.npy")[:, :, :20]
X_daily = np.load("data/multitask/X_daily.npy")
y_entry = np.load("data/multitask/y_entry.npy")
y_direction = np.load("data/multitask/y_direction.npy")

# ✅ entry=1 필터링
mask = y_entry == 1
X_minute = X_minute[mask]
X_daily = X_daily[mask]
y_direction = y_direction[mask]

# ✅ 텐서로 변환
x_min_tensor = torch.tensor(X_minute, dtype=torch.float32)
x_day_tensor = torch.tensor(X_daily, dtype=torch.float32)
y_tensor = torch.tensor(y_direction, dtype=torch.long)

dataset = TensorDataset(x_min_tensor, x_day_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# ✅ 모델 로딩
model = DualBranchLSTM(input_size_minute=20, input_size_daily=23)
model.load_state_dict(torch.load("saved_direction_only_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ✅ 추론
all_preds = []
with torch.no_grad():
    for xb_min, xb_day, _ in loader:
        xb_min = xb_min.to(DEVICE)
        xb_day = xb_day.to(DEVICE)
        _, preds = model(xb_min, xb_day)
        all_preds.append(preds.cpu())

# ✅ 평가
all_preds = torch.cat(all_preds).numpy()
pred_labels = all_preds.argmax(axis=1)

print("🎯 entry=1 샘플 수:", len(y_direction))
print("🔍 [Direction 예측 성능]")
print(confusion_matrix(y_direction, pred_labels))
print(classification_report(y_direction, pred_labels, digits=4))
print(f"✅ Accuracy: {accuracy_score(y_direction, pred_labels):.4f}")
