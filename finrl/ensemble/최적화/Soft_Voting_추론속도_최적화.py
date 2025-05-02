import numpy as np

# ✅ 세 모델 예측 결과 불러오기
preds_lstm = np.load("lstm_preds.npy", allow_pickle=True).item()
preds_lightgbm = np.load("lightgbm_preds.npy", allow_pickle=True).item()
preds_tabnet = np.load("tabnet_preds.npy", allow_pickle=True).item()

entry_probs_lstm = preds_lstm['entry_probs']
direction_probs_lstm = preds_lstm['direction_probs']

entry_probs_lightgbm = preds_lightgbm['entry_probs']
direction_probs_lightgbm = preds_lightgbm['direction_probs']

entry_probs_tabnet = preds_tabnet['entry_probs']
direction_probs_tabnet = preds_tabnet['direction_probs']

# ✅ 최적화된 Soft Voting 계산
entry_stack = np.vstack([entry_probs_lstm, entry_probs_lightgbm, entry_probs_tabnet])
direction_stack = np.vstack([direction_probs_lstm, direction_probs_lightgbm, direction_probs_tabnet])

final_entry_probs = np.mean(entry_stack, axis=0)
final_direction_probs = np.mean(direction_stack, axis=0)

# ✅ Threshold 적용 (벡터화)
pred_entry = (final_entry_probs > 0.5).astype(int)
pred_direction = (final_direction_probs > 0.5).astype(int)

# ✅ 결과 확인 (상위 10개 샘플)
print("===== 최적화된 Soft Voting 예측 결과 (Top 10) =====")
for i in range(10):
    print(f"[{i}] Entry Prob: {final_entry_probs[i]:.4f}, Direction Prob: {final_direction_probs[i]:.4f}")
    print(f"     Entry Predict: {pred_entry[i]}, Direction Predict: {pred_direction[i]}")
