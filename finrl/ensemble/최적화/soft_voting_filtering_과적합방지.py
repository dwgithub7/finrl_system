import numpy as np

# ✅ 예시: 모델별 Validation Accuracy (가중치 근거)
validation_scores = {
    'lstm': 0.58,
    'lightgbm': 0.52,
    'tabnet': 0.60
}

# ✅ Validation Accuracy 기반 Weight 계산
scores = np.array(list(validation_scores.values()))
weights = scores / np.sum(scores)  # 정규화해서 합 = 1
model_names = list(validation_scores.keys())

print("✅ 모델별 가중치:")
for name, weight in zip(model_names, weights):
    print(f"{name}: {weight:.4f}")

# ✅ 모델별 예측 결과 로드
all_preds = {
    'lstm': np.load("lstm_preds.npy", allow_pickle=True).item(),
    'lightgbm': np.load("lightgbm_preds.npy", allow_pickle=True).item(),
    'tabnet': np.load("tabnet_preds.npy", allow_pickle=True).item()
}

# ✅ 가중 합 Soft Voting 계산
entry_ensemble = np.zeros_like(all_preds['lstm']['entry_probs'])
direction_ensemble = np.zeros_like(all_preds['lstm']['direction_probs'])

for model_name, weight in zip(model_names, weights):
    entry_ensemble += all_preds[model_name]['entry_probs'] * weight
    direction_ensemble += all_preds[model_name]['direction_probs'] * weight

# ✅ Threshold 적용 (0.5)
pred_entry = (entry_ensemble > 0.5).astype(int)
pred_direction = (direction_ensemble > 0.5).astype(int)

# ✅ 결과 확인 (상위 10개)
print("\n===== Weighted Soft Voting 결과 (Top 10) =====")
for i in range(10):
    print(f"[{i}] Entry Prob: {entry_ensemble[i]:.4f}, Direction Prob: {direction_ensemble[i]:.4f}")
    print(f"     Entry Predict: {pred_entry[i]}, Direction Predict: {pred_direction[i]}")
