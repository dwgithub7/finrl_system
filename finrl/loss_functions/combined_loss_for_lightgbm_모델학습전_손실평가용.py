import numpy as np
import os
from sklearn.metrics import log_loss
from datetime import datetime
from sklearn.model_selection import train_test_split

from finrl.training.train_lightgbm_models import LightGBMTrainer
from finrl.preprocessing.feature_generator import load_dualbranch_data, generate_combined_features, filter_by_entry

# ✅ 손실 함수 정의
def binary_focal_loss(probs, targets, gamma=2.0, eps=1e-6):
    probs = np.clip(probs, eps, 1 - eps)
    loss = - (1 - probs) ** gamma * (targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
    return np.mean(loss)

def softmax_focal_loss(probs, targets, gamma=2.0, eps=1e-6):
    probs = np.clip(probs, eps, 1 - eps)
    loss = - (1 - probs) ** gamma * (targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
    return np.mean(loss)

def combined_loss(entry_probs, entry_true, direction_probs, direction_true, alpha=1.0, beta=2.0):
    entry_loss = binary_focal_loss(entry_probs, entry_true)
    direction_loss = softmax_focal_loss(direction_probs, direction_true)
    total_loss = alpha * entry_loss + beta * direction_loss
    return total_loss, entry_loss, direction_loss

# ✅ 데이터 로딩 및 학습
if __name__ == "__main__":
    DATA_DIR = "data/dualbranch/"
    os.makedirs("testlog", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"testlog/combined_loss_for_lightgbm_log_{timestamp}.txt"

    X_minute, X_daily, y_entry, y_direction = load_dualbranch_data(DATA_DIR)
    X_features = generate_combined_features(X_minute, X_daily)
    x_dir, _, y_dir_filtered = filter_by_entry(X_features, y_entry, y_direction)

    # 훈련 및 예측
    trainer = LightGBMTrainer(
        x_entry=X_features,
        y_entry=y_entry,
        x_direction=x_dir,
        y_direction=y_dir_filtered
    )
    trainer.train()
    entry_probs, direction_probs = trainer.predict(X_features)

    # ✅ Combined Loss 계산 및 로깅
    with open(log_path, "w") as f:
        total, entry, direction = combined_loss(entry_probs, y_entry, direction_probs, y_direction)
        log_text = f"Total: {total:.4f}, Entry: {entry:.4f}, Direction: {direction:.4f}"
        print(log_text)
        f.write(log_text + "\n")
