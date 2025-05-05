import os
import numpy as np
import torch
from datetime import datetime
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split

from finrl.preprocessing.feature_generator import generate_combined_features, filter_by_entry, load_dualbranch_data

# ✅ Focal Loss 정의
def binary_focal_loss(probs, targets, gamma=2.0, eps=1e-8):
    probs = np.clip(probs, eps, 1. - eps)
    loss = - (1 - probs) ** gamma * (targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
    return np.mean(loss)

def softmax_focal_loss(probs, targets, gamma=2.0, eps=1e-8):
    probs = np.clip(probs, eps, 1. - eps)
    targets_onehot = np.eye(2)[targets.astype(int)]
    loss = - (1 - probs) ** gamma * targets_onehot * np.log(probs)
    return np.mean(np.sum(loss, axis=1))

# ✅ 데이터 로딩
DATA_DIR = "data/dualbranch"
X_minute, X_daily, y_entry, y_direction = load_dualbranch_data(DATA_DIR)
X_features = generate_combined_features(X_minute, X_daily)

# ✅ Entry용 학습/검증 데이터 분할
X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(
    X_features, y_entry, test_size=0.2, random_state=42
)

# ✅ Direction용 데이터는 Entry==1인 것만 추출
X_dir, _, y_dir = filter_by_entry(X_features, y_entry, y_direction)
X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(
    X_dir, y_dir, test_size=0.2, random_state=42
)

# ✅ TabNet 모델 생성 함수
def build_tabnet():
    return TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0
    )

# ✅ 모델 인스턴스 생성 및 학습
entry_model = build_tabnet()
direction_model = build_tabnet()

print(" Training Entry model...")
entry_model.fit(
    X_train=X_train_e, y_train=y_train_e,
    eval_set=[(X_val_e, y_val_e)],
    eval_name=['val'],
    eval_metric=['logloss'],
    patience=20,
    batch_size=4096,
    virtual_batch_size=1024,
    num_workers=0,
    drop_last=False
)

print(" Training Direction model...")
direction_model.fit(
    X_train=X_train_d, y_train=y_train_d,
    eval_set=[(X_val_d, y_val_d)],
    eval_name=['val'],
    eval_metric=['logloss'],
    patience=20,
    batch_size=4096,
    virtual_batch_size=1024,
    num_workers=0,
    drop_last=False
)

# ✅ 예측 및 손실 계산
entry_probs = entry_model.predict_proba(X_val_e)[:, 1]
direction_probs = direction_model.predict_proba(X_val_d)

entry_loss = binary_focal_loss(entry_probs, y_val_e)
direction_loss = softmax_focal_loss(direction_probs, y_val_d)
total_loss = entry_loss + 2 * direction_loss

# ✅ 결과 로그 출력
log_dir = "testlog"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"combined_loss_for_tabnet_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

with open(log_path, "w") as f:
    f.write(f"Total: {total_loss:.4f}, Entry: {entry_loss:.4f}, Direction: {direction_loss:.4f}\n")
    print(f"Total: {total_loss:.4f}, Entry: {entry_loss:.4f}, Direction: {direction_loss:.4f}")
