
import numpy as np
import torch
from lstm_dualbranch_model import DualBranchLSTM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ 설정값
DATA_DIR = "data/multitask/"
MODEL_PATH = "saved_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 데이터 로딩
X_minute = np.load(DATA_DIR + "realtest_X_minute.npy")
X_daily = np.load(DATA_DIR + "realtest_X_daily.npy")
y_entry = np.load(DATA_DIR + "realtest_y_entry.npy")
y_direction = np.load(DATA_DIR + "realtest_y_direction.npy")

# 🔹 모델 준비
input_size_min = X_minute.shape[2]
input_size_day = X_daily.shape[2]
model = DualBranchLSTM(input_size_minute=20, input_size_daily=23)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 🔹 추론
y_entry_pred, y_dir_pred = [], []
with torch.no_grad():
    for i in range(len(X_minute)):
        x_min = torch.tensor(X_minute[i:i+1], dtype=torch.float32).to(DEVICE)
        x_day = torch.tensor(X_daily[i:i+1], dtype=torch.float32).to(DEVICE)
        entry_prob, dir_prob = model(x_min, x_day)
        entry = int(entry_prob.item() > 0.5)
        direction = int(dir_prob.argmax(dim=1).item())
        y_entry_pred.append(entry)
        y_dir_pred.append(direction)

# 🔹 평가 (진입한 샘플만)
y_entry_pred = np.array(y_entry_pred)
y_dir_pred = np.array(y_dir_pred)

mask = y_entry_pred == 1
y_true = y_direction[mask]
y_pred = y_dir_pred[mask]

print(f"🎯 진입 샘플 수: {len(y_true)} / 전체: {len(y_entry)}")
print("🔍 [Direction 예측 성능 - 진입 샘플만]")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
