# ✅ Soft Voting Ensemble (실전형 개선버전)

import os
import numpy as np
import torch

from finrl.training.train_dualbranch_lstm_v4_final import DualBranchLSTMTrainer
from finrl.training.train_lightgbm_models import LightGBMTrainer
from finrl.training.train_tabnet_models import TabNetTrainer

def predict_in_batches(trainer, X_minute, X_daily, batch_size=512):
    trainer.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entry_probs = []
    direction_probs = []

    total = X_minute.shape[0]
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)

        x_minute_batch = torch.tensor(X_minute[start_idx:end_idx], dtype=torch.float32).to(device)
        x_daily_batch = torch.tensor(X_daily[start_idx:end_idx], dtype=torch.float32).to(device)

        with torch.no_grad():
            entry_logits, direction_logits = trainer.model(x_minute_batch, x_daily_batch)
            entry_probs.append(torch.sigmoid(entry_logits).squeeze(-1).cpu().numpy())
            direction_probs.append(torch.softmax(direction_logits, dim=-1)[:, 1].cpu().numpy())

    entry_probs = np.concatenate(entry_probs, axis=0)
    direction_probs = np.concatenate(direction_probs, axis=0)
    return entry_probs, direction_probs


# ✅ 데이터 로드
DATA_DIR = "data/dualbranch/"

X_minute = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_X_minute_full.npy"))
X_daily = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_X_daily_full.npy"))

# ✅ 1분봉과 일봉 각각 마지막 timestep만 추출
X_minute_last = X_minute[:, -1, :]  # (batch, feature_dim_minute)
X_daily_last = X_daily[:, -1, :]    # (batch, feature_dim_daily)

# ✅ Tabular 데이터 생성 (LightGBM/TabNet용)
X_features = np.concatenate([X_minute_last, X_daily_last], axis=1).astype(np.float32)  # (batch, total_feature_dim)

# ✅ LSTM 모델 로드 (DualBranch Bidirectional Attention LSTM)
lstm_trainer = DualBranchLSTMTrainer(input_size_minute=17, input_size_daily=20)
lstm_trainer.load(model_name="dualbranch_lstm")

# ✅ LightGBM 모델 로드
lightgbm_trainer = LightGBMTrainer(feature_dim=X_features.shape[1])
lightgbm_trainer.load()

# ✅ TabNet 모델 로드
tabnet_trainer = TabNetTrainer(feature_dim=X_features.shape[1])
tabnet_trainer.load()

# ✅ LSTM 예측 (Entry / Direction)
entry_probs_lstm, direction_probs_lstm = predict_in_batches(lstm_trainer, X_minute, X_daily, batch_size=512)

# ✅ LightGBM 예측
entry_probs_lightgbm, direction_probs_lightgbm = lightgbm_trainer.predict(X_features)

# ✅ TabNet 예측
entry_probs_tabnet, direction_probs_tabnet = tabnet_trainer.predict(X_features)

# ✅ Soft Voting (예측 확률 평균)
final_entry_probs = (entry_probs_lstm + entry_probs_lightgbm + entry_probs_tabnet) / 3
final_direction_probs = (direction_probs_lstm + direction_probs_lightgbm + direction_probs_tabnet) / 3

# ✅ 결과 출력
print("\n✅ Soft Voting Ensemble 결과 (상위 10개 샘플)")
for i in range(10):
    print(f"[{i}] Entry Prob: {final_entry_probs[i]:.4f}, Direction Prob: {final_direction_probs[i]:.4f}")

# ✅ Threshold 기반 신호 해석
ENTRY_THRESHOLD = 0.5





print("\n✅ 매수/매도 판단 예시 (Entry > 0.5 기준)")
for i in range(10):
    entry_signal = "진입" if final_entry_probs[i] > ENTRY_THRESHOLD else "관망"
    direction_signal = "롱" if final_direction_probs[i] > 0.5 else "숏"
    print(f"[{i}] Entry: {entry_signal}, Direction: {direction_signal}")

# ✅ (선택) 결과 저장
np.save(os.path.join(DATA_DIR, "ensemble_entry_probs.npy"), final_entry_probs)
np.save(os.path.join(DATA_DIR, "ensemble_direction_probs.npy"), final_direction_probs)
