
import torch
import numpy as np
from lstm_dualbranch_model import DualBranchLSTM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 디바이스:", device)

# ✅ 데이터 로딩 및 피처 자르기
X_minute = np.load("data/multitask/X_minute.npy")[:, :, :20]  # 🔧 앞의 20개 피처만 사용
X_daily = np.load("data/multitask/X_daily.npy")
y_entry = np.load("data/multitask/y_entry.npy")
y_direction = np.load("data/multitask/y_direction.npy")

# ✅ 모델 정의 (input 크기 맞춤)
model = DualBranchLSTM(input_size_minute=20, input_size_daily=23)
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

# ▶ 진입 예측이 1인 샘플만 평가
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
