import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openpyxl import Workbook

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

# ✅ 5. 엑셀 파일로 저장
wb = Workbook()
ws = wb.active
ws.title = "Soft Voting Evaluation"

# 헤더 작성
ws.append(["평가 항목", "Entry 예측", "Direction 예측"])

# 데이터 작성
ws.append(["Accuracy", f"{entry_accuracy*100:.2f}%", f"{direction_accuracy*100:.2f}%"])
ws.append(["Precision", f"{entry_precision*100:.2f}%", f"{direction_precision*100:.2f}%"])
ws.append(["Recall", f"{entry_recall*100:.2f}%", f"{direction_recall*100:.2f}%"])
ws.append(["F1 Score", f"{entry_f1*100:.2f}%", f"{direction_f1*100:.2f}%"])

# 파일 저장
report_path = "soft_voting_eval_report.xlsx"
wb.save(report_path)

print(f"✅ Soft Voting 평가 결과가 {report_path} 로 저장되었습니다.")
