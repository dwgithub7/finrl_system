# ✅ 완성형 eval_dualbranch.py (진입+방향 평가 모두)

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# ✅ 설정값
DATA_DIR = "data/dualbranch"
MODEL_PATH = "data/models/dualbranch_model.pth"
SYMBOL_PERIOD = "SOLUSDT_1m_finrl_20240101~20250426"

# entry=1 ➔ "매매 기회 있음" ➔ direction 예측 필요 (1: 매수, 0: 매도)
# entry=0 ➔ "관망 (아무것도 하지 않음)" ➔ direction 예측 무의미
EVAL_MODE = "entry_only"  # "entry_only" 또는 "all", all은 진입과 관망 모두 평가

BATCH_SIZE = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 평가 디바이스: {DEVICE}")

# ✅ 데이터 로딩
X_minute = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_X_minute.npy"))
X_daily = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_X_daily.npy"))
y_entry = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_y_entry.npy"))
y_direction = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_y_direction.npy"))

# ✅ 모델 정의 (구조는 학습 때와 같아야 함)
class DualBranchLSTM(torch.nn.Module):
    def __init__(self, input_size_minute, input_size_daily, hidden_size=128):
        super().__init__()
        self.lstm_minute = torch.nn.LSTM(input_size_minute, hidden_size, batch_first=True)
        self.lstm_daily = torch.nn.LSTM(input_size_daily, hidden_size, batch_first=True)
        self.fc_entry = torch.nn.Linear(hidden_size * 2, 1)  # 진입 예측
        self.fc_direction = torch.nn.Linear(hidden_size * 2, 2)  # 방향 예측
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_minute, x_daily):
        _, (h_minute, _) = self.lstm_minute(x_minute)
        _, (h_daily, _) = self.lstm_daily(x_daily)
        h_combined = torch.cat((h_minute[-1], h_daily[-1]), dim=1)
        entry_out = self.sigmoid(self.fc_entry(h_combined))
        direction_out = self.fc_direction(h_combined)
        return entry_out, direction_out

# ✅ 모델 로드
input_size_minute = X_minute.shape[2]
input_size_daily = X_daily.shape[2]
model = DualBranchLSTM(input_size_minute, input_size_daily)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ✅ 데이터 변환
X_minute_tensor = torch.tensor(X_minute, dtype=torch.float32).to(DEVICE)
X_daily_tensor = torch.tensor(X_daily, dtype=torch.float32).to(DEVICE)

# ✅ 배치 평가
entry_preds = []
dir_preds = []

with torch.no_grad():
    for i in range(0, len(X_minute_tensor), BATCH_SIZE):
        x_min_batch = X_minute_tensor[i:i+BATCH_SIZE]
        x_day_batch = X_daily_tensor[i:i+BATCH_SIZE]

        entry_logit, dir_logit = model(x_min_batch, x_day_batch)

        entry_pred = (entry_logit > 0.5).float().cpu().numpy().squeeze()
        dir_pred = torch.argmax(dir_logit, dim=1).cpu().numpy()

        entry_preds.append(entry_pred)
        dir_preds.append(dir_pred)

entry_pred_label = np.concatenate(entry_preds)
dir_pred_label = np.concatenate(dir_preds)

# ✅ 평가할 데이터 선택
if EVAL_MODE == "entry_only":
    mask = entry_pred_label == 1
    y_true = y_direction[mask]
    y_pred = dir_pred_label[mask]
    print(f"🎯 진입=1 샘플만 평가: {len(y_true)}개")
else:
    y_true = y_direction
    y_pred = dir_pred_label
    print(f"🎯 전체 데이터 평가: {len(y_true)}개")

# ✅ 평가 출력
print("\n🔍 Confusion Matrix")
print(confusion_matrix(y_true, y_pred))
print("\n🔍 Classification Report")
print(classification_report(y_true, y_pred, digits=4))
print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
