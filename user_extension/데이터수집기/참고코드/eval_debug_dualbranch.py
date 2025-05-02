
import torch
import numpy as np
from lstm_dualbranch_model import DualBranchLSTM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 디바이스:", device)

# ✅ 데이터 로딩
X_minute = np.load("data/multitask/debug_X_minute.npy")
X_daily = np.load("data/multitask/debug_X_daily.npy")
y_entry = np.load("data/multitask/debug_y_entry.npy")
y_direction = np.load("data/multitask/debug_y_direction.npy")

# ✅ 모델 준비
input_size_min = X_minute.shape[2]
input_size_day = X_daily.shape[2]
model = DualBranchLSTM(input_size_min, input_size_day)
model.load_state_dict(torch.load("saved_model.pt", map_location=device))
model.to(device)
model.eval()

# ✅ 추론
with torch.no_grad():
    x_min = torch.tensor(X_minute, dtype=torch.float32).to(device)
    x_day = torch.tensor(X_daily, dtype=torch.float32).to(device)
    entry_pred, dir_pred = model(x_min, x_day)

# ✅ 평가
entry_pred_bin = (entry_pred.cpu().numpy() > 0.5).astype(int).flatten()
dir_pred_class = dir_pred.argmax(dim=1).cpu().numpy()

# ▶ 평가 기준: entry == 1인 샘플만 대상
mask = entry_pred_bin == 1
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
