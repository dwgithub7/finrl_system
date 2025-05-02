import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# ✅ 설정값
DATA_DIR = "data/testset_dualbranch"
MODEL_PATH = "data/models/dualbranch_model.pth"  # 학습 완료된 모델
SYMBOL_PERIOD = "SOLUSDT_1m_finrl_20240101~20250426"
EVAL_MODE = "entry_only"  # "entry_only" 또는 "all"
BATCH_SIZE = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ 평가 디바이스: {DEVICE}")

# ✅ 데이터 로딩
X_minute = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_X_minute_test.npy"))
X_daily = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_X_daily_test.npy"))
y_entry = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_y_entry_test.npy"))
y_direction = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_y_direction_test.npy"))

# ✅ 모델 정의 (학습할 때와 동일해야 함)
class DualBranchLSTM(torch.nn.Module):
    def __init__(self, input_size_minute, input_size_daily, hidden_size=128):
        super().__init__()
        self.lstm_minute = torch.nn.LSTM(input_size_minute, hidden_size, batch_first=True)
        self.lstm_daily = torch.nn.LSTM(input_size_daily, hidden_size, batch_first=True)
        self.fc_entry = torch.nn.Linear(hidden_size * 2, 1)
        self.fc_direction = torch.nn.Linear(hidden_size * 2, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_minute, x_daily):
        _, (h_minute, _) = self.lstm_minute(x_minute)
        _, (h_daily, _) = self.lstm_daily(x_daily)
        h_combined = torch.cat((h_minute[-1], h_daily[-1]), dim=1)
        entry_logit = self.sigmoid(self.fc_entry(h_combined)).squeeze(1)
        dir_logit = self.fc_direction(h_combined)
        return entry_logit, dir_logit

# ✅ 모델 로드
input_size_minute = X_minute.shape[2]
input_size_daily = X_daily.shape[2]
model = DualBranchLSTM(input_size_minute, input_size_daily)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ✅ 배치 단위 평가
all_entry_preds = []
all_dir_preds = []

with torch.no_grad():
    for start_idx in range(0, len(X_minute), BATCH_SIZE):
        end_idx = start_idx + BATCH_SIZE
        x_min_batch = torch.tensor(X_minute[start_idx:end_idx], dtype=torch.float32).to(DEVICE)
        x_day_batch = torch.tensor(X_daily[start_idx:end_idx], dtype=torch.float32).to(DEVICE)

        entry_logit, dir_logit = model(x_min_batch, x_day_batch)
        entry_pred_label = (entry_logit > 0.5).long().cpu().numpy()
        dir_pred_label = torch.argmax(dir_logit, dim=1).cpu().numpy()

        all_entry_preds.append(entry_pred_label)
        all_dir_preds.append(dir_pred_label)

entry_preds = np.concatenate(all_entry_preds)
dir_preds = np.concatenate(all_dir_preds)

# ✅ 평가할 데이터 선택
if EVAL_MODE == "entry_only":
    mask = entry_preds == 1
    y_true = y_direction[mask]
    y_pred = dir_preds[mask]
    print(f"\n🎯 진입=1 샘플만 평가: {len(y_true)}개")
else:
    y_true = y_direction
    y_pred = dir_preds
    print(f"\n🎯 전체 데이터 평가: {len(y_true)}개")

# ✅ 결과 출력
print("\n🔍 Confusion Matrix")
print(confusion_matrix(y_true, y_pred))
print("\n🔍 Classification Report")
print(classification_report(y_true, y_pred, digits=4))
print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
