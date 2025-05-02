import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ✅ 1. Soft Voting 결과 불러오기
preds = np.load("soft_voting_preds.npy", allow_pickle=True).item()
entry_probs = preds['entry_probs']
direction_probs = preds['direction_probs']

# ✅ 2. 실제 정답 데이터 불러오기
y_true_entry = np.load("y_true_entry.npy")
y_true_direction = np.load("y_true_direction.npy")

# ✅ 3. Threshold 적용 (0.5 기준)
pred_entry = (entry_probs > 0.5).astype(int)
pred_direction = (direction_probs > 0.5).astype(int)

# ✅ 4. 평가 수행
entry_accuracy = accuracy_score(y_true_entry, pred_entry)
entry_precision = precision_score(y_true_entry, pred_entry)
entry_recall = recall_score(y_true_entry, pred_entry)
entry_f1 = f1_score(y_true_entry, pred_entry)

direction_accuracy = accuracy_score(y_true_direction, pred_direction)
direction_precision = precision_score(y_true_direction, pred_direction)
direction_recall = recall_score(y_true_direction, pred_direction)
direction_f1 = f1_score(y_true_direction, pred_direction)

# ✅ 5. 결과 출력
print("===== Entry Prediction 평가 (실제 데이터 기준) =====")
print(f"Accuracy: {entry_accuracy*100:.2f}%")
print(f"Precision: {entry_precision*100:.2f}%")
print(f"Recall: {entry_recall*100:.2f}%")
print(f"F1 Score: {entry_f1*100:.2f}%")

print("\n===== Direction Prediction 평가 (실제 데이터 기준) =====")
print(f"Accuracy: {direction_accuracy*100:.2f}%")
print(f"Precision: {direction_precision*100:.2f}%")
print(f"Recall: {direction_recall*100:.2f}%")
print(f"F1 Score: {direction_f1*100:.2f}%")
