import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# ✅ 설정값
DATA_DIR = "data/testset_dualbranch"  # 테스트셋 저장 위치
MODEL_PATH = "data/models/dualbranch_model.pth"  # 저장된 학습 모델
SYMBOL_PERIOD = "SOLUSDT_1m_finrl_20240101~20250426"
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ 평가 디바이스: {DEVICE}")

# ✅ 테스트셋 로딩
X_minute = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_X_minute_test.npy"))
X_daily = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_X_daily_test.npy"))
y_entry = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_y_entry_test.npy"))
y_direction = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_y_direction_test.npy"))

# ✅ 모델 구조 (학습 때와 동일하게 맞춰야 함)
class DualBranchLSTM(torch.nn.Module):
    def __init__(self, input_size_minute, input_size_daily, hidden_size=128):
        super().__init__()
        self.lstm_minute = torch.nn.LSTM(input_size_minute, hidden_size, batch_first=True)
        self.lstm_daily = torch.nn.LSTM(input_size_daily, hidden_size, batch_first=True)
        self.fc_entry = torch.nn.Linear(hidden_size * 2, 1)  # 관망/진입 판단
        self.fc_direction = torch.nn.Linear(hidden_size * 2, 2)  # 매수/매도 방향 판단
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_minute, x_daily):
        _, (h_minute, _) = self.lstm_minute(x_minute)
        _, (h_daily, _) = self.lstm_daily(x_daily)
        h_combined = torch.cat((h_minute[-1], h_daily[-1]), dim=1)
        entry_logit = self.fc_entry(h_combined)
        dir_logit = self.fc_direction(h_combined)
        return entry_logit, dir_logit

# ✅ 모델 불러오기
input_size_minute = X_minute.shape[2]
input_size_daily = X_daily.shape[2]
model = DualBranchLSTM(input_size_minute, input_size_daily)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ✅ 배치 평가
X_minute_tensor = torch.tensor(X_minute, dtype=torch.float32)
X_daily_tensor = torch.tensor(X_daily, dtype=torch.float32)

entry_preds = []
dir_preds = []

for i in range(0, len(X_minute_tensor), BATCH_SIZE):
    batch_minute = X_minute_tensor[i:i+BATCH_SIZE].to(DEVICE)
    batch_daily = X_daily_tensor[i:i+BATCH_SIZE].to(DEVICE)
    with torch.no_grad():
        entry_logit, dir_logit = model(batch_minute, batch_daily)
        entry_pred_label = (torch.sigmoid(entry_logit).squeeze() > 0.5).long().cpu().numpy()
        dir_pred_label = torch.argmax(dir_logit, dim=1).cpu().numpy()
        entry_preds.append(entry_pred_label)
        dir_preds.append(dir_pred_label)

entry_preds = np.concatenate(entry_preds)
dir_preds = np.concatenate(dir_preds)

# ✅ 진입 샘플(=entry=1)만 평가
mask = entry_preds == 1
y_true = y_direction[mask]
y_pred = dir_preds[mask]

print(f"\n🎯 테스트셋 진입=1 샘플 수: {len(y_true)}")

# ✅ 평가 출력
print("\n🔍 Confusion Matrix")
print(confusion_matrix(y_true, y_pred))
print("\n🔍 Classification Report")
print(classification_report(y_true, y_pred, digits=4))
print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
