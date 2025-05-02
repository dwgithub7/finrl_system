
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from lstm_dualbranch_model import DualBranchLSTM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 디바이스:", device)

# ✅ 데이터 로딩 및 피처 자르기
X_minute = np.load("data/multitask/X_minute.npy")[:, :, :20]
X_daily = np.load("data/multitask/X_daily.npy")
y_entry = np.load("data/multitask/y_entry.npy")
y_direction = np.load("data/multitask/y_direction.npy")

# ✅ 텐서로 변환
x_min_tensor = torch.tensor(X_minute, dtype=torch.float32)
x_day_tensor = torch.tensor(X_daily, dtype=torch.float32)
entry_tensor = torch.tensor(y_entry, dtype=torch.float32)
dir_tensor = torch.tensor(y_direction, dtype=torch.long)

# ✅ DataLoader 정의 (배치 사이즈 조절 가능)
BATCH_SIZE = 2048
dataset = TensorDataset(x_min_tensor, x_day_tensor, entry_tensor, dir_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# ✅ 모델 정의 및 로드
model = DualBranchLSTM(input_size_minute=20, input_size_daily=23)
model.load_state_dict(torch.load("saved_model.pt", map_location=device))
model.to(device)
model.eval()

# ✅ 추론 수행
entry_preds = []
dir_preds = []

with torch.no_grad():
    for xb_min, xb_day, _, _ in loader:
        xb_min = xb_min.to(device)
        xb_day = xb_day.to(device)
        entry_pred, dir_pred = model(xb_min, xb_day)
        entry_preds.append(entry_pred.cpu())
        dir_preds.append(dir_pred.cpu())

# ✅ 결과 정리 및 평가
entry_preds = torch.cat(entry_preds).numpy()
dir_preds = torch.cat(dir_preds).numpy()

entry_bin = (entry_preds > 0.5).astype(int).flatten()
dir_pred_class = dir_preds.argmax(axis=1)

mask = entry_bin == 1
if mask.sum() > 0:
    y_true = y_direction[mask]
    y_pred = dir_pred_class[mask]

    print("🎯 진입 샘플 수:", len(y_true))
    print("🔍 [Direction 예측 성능 - 진입 샘플만]")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
    print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
else:
    print("⚠️ 예측 결과에서 진입 샘플 없음")
