import numpy as np

# ✅ 예시: 모델별 Validation Accuracy 준비
validation_scores = {
    'lstm': 0.58,  # 예시로 LSTM 모델의 Validation Accuracy
    'lightgbm': 0.52,  # LightGBM 모델 Validation Accuracy
    'tabnet': 0.60  # TabNet 모델 Validation Accuracy
}

# ✅ 기준값 설정 (Validation Accuracy 0.55 이상 모델만 포함)
threshold_accuracy = 0.55

# ✅ Soft Voting 대상 모델 결정
selected_models = [model for model, score in validation_scores.items() if score >= threshold_accuracy]
print(f"✅ Soft Voting에 포함될 모델: {selected_models}")

# ✅ 모델별 예측 결과 로드 (예시)
all_preds = {
    'lstm': np.load("lstm_preds.npy", allow_pickle=True).item(),
    'lightgbm': np.load("lightgbm_preds.npy", allow_pickle=True).item(),
    'tabnet': np.load("tabnet_preds.npy", allow_pickle=True).item()
}

# ✅ Soft Voting에 사용할 entry_probs, direction_probs 수집
entry_list = []
direction_list = []

for model_name in selected_models:
    entry_list.append(all_preds[model_name]['entry_probs'])
    direction_list.append(all_preds[model_name]['direction_probs'])

# ✅ Soft Voting 평균 계산
entry_stack = np.vstack(entry_list)
direction_stack = np.vstack(direction_list)

final_entry_probs = np.mean(entry_stack, axis=0)
final_direction_probs = np.mean(direction_stack, axis=0)

# ✅ Threshold 적용 (0.5 기준)
pred_entry = (final_entry_probs > 0.5).astype(int)
pred_direction = (final_direction_probs > 0.5).astype(int)

# ✅ 결과 확인 (상위 10개)
print("\n===== 최적화된 Soft Voting 결과 (Top 10) =====")
for i in range(10):
    print(f"[{i}] Entry Prob: {final_entry_probs[i]:.4f}, Direction Prob: {final_direction_probs[i]:.4f}")
    print(f"     Entry Predict: {pred_entry[i]}, Direction Predict: {pred_direction[i]}")
