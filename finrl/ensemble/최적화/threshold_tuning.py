import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ✅ Soft Voting 예측 결과 불러오기 (예시)
preds = np.load("soft_voting_preds.npy", allow_pickle=True).item()
entry_probs = preds['entry_probs']

# ✅ 실제 정답 불러오기
y_true_entry = np.load("y_true_entry.npy")

# ✅ Threshold 튜닝 범위 설정
thresholds = np.arange(0.4, 0.6, 0.01)

precision_list = []
recall_list = []
f1_list = []

# ✅ 각 Threshold에 대해 Precision, Recall, F1 계산
for threshold in thresholds:
    pred_entry = (entry_probs > threshold).astype(int)
    precision = precision_score(y_true_entry, pred_entry)
    recall = recall_score(y_true_entry, pred_entry)
    f1 = f1_score(y_true_entry, pred_entry)

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

# ✅ 최적 Threshold 찾기
best_idx = np.argmax(f1_list)
best_threshold = thresholds[best_idx]

print("===== Threshold 튜닝 결과 =====")
print(f"최적 Threshold: {best_threshold:.2f}")
print(f"Precision: {precision_list[best_idx]*100:.2f}%")
print(f"Recall: {recall_list[best_idx]*100:.2f}%")
print(f"F1 Score: {f1_list[best_idx]*100:.2f}%")

# ✅ 그래프 시각화
plt.figure(figsize=(8,6))
plt.plot(thresholds, precision_list, label="Precision", marker='o')
plt.plot(thresholds, recall_list, label="Recall", marker='o')
plt.plot(thresholds, f1_list, label="F1 Score", marker='o')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision/Recall/F1 vs Threshold")
plt.legend()
plt.grid()
plt.show()
