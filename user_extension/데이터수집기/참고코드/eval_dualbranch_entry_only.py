
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ 경로 및 파일명 (dualbranch용)
DATA_DIR = "data/multitask/"
y_entry = np.load(DATA_DIR + "y_entry.npy")
y_direction = np.load(DATA_DIR + "y_direction.npy")

# 예시: 예측 결과 불러오기
# 아래는 테스트용 더미 (실전에서는 추론 결과로 교체)
y_entry_pred = y_entry.copy()
y_direction_pred = y_direction.copy()

# ✅ 진입 샘플만 추출
mask = y_entry_pred == 1
y_true = y_direction[mask]
y_pred = y_direction_pred[mask]

print(f"🎯 진입 샘플 수: {len(y_true)} / 전체: {len(y_entry)}")

# ✅ 평가 결과 출력
print("🔍 [Direction 예측 성능 - 진입 샘플만]")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
