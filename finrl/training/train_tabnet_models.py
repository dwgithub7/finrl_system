# ✅ 수정 완료된 train_tabnet_models_v2.py (실전용)

import os
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
from tqdm import tqdm
from pytorch_tabnet.callbacks import Callback

# ✅ TabNet TQDM Progress Callback
class TQDMCallback(Callback):
    def __init__(self, total_epochs=200):
        super().__init__()
        self.pbar = tqdm(total=total_epochs, desc="TabNet Training")

    def on_epoch_end(self, epoch_idx, logs=None):
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()

# ✅ TabNet Trainer Class
class TabNetTrainer:
    def __init__(self, feature_dim, model_dir="models"):
        self.feature_dim = feature_dim
        self.model_dir = model_dir
        self.entry_model = None
        self.direction_model = None

    def train(self, X, entry_y, direction_y):
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        # ✅ 수정된 올바른 split
        X_train, X_val, entry_y_train, entry_y_val, direction_y_train, direction_y_val = train_test_split(
            X, entry_y, direction_y, test_size=0.2, random_state=42
        )

        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)

        # Train Entry model
        self.entry_model = self._create_tabnet_model()
        print("Training Entry model...")
        self.entry_model.fit(
            X_train=X_train, y_train=entry_y_train,
            eval_set=[(X_val, entry_y_val)],
            eval_name=['val'],
            eval_metric=['logloss'],
            patience=20,
            batch_size=4096,
            virtual_batch_size=1024,
            num_workers=0,
            drop_last=False,
            callbacks=[TQDMCallback(total_epochs=200)]
        )

        # Train Direction model
        self.direction_model = self._create_tabnet_model()
        print("Training Direction model...")
        self.direction_model.fit(
            X_train=X_train, y_train=direction_y_train,
            eval_set=[(X_val, direction_y_val)],
            eval_name=['val'],
            eval_metric=['logloss'],
            patience=20,
            batch_size=4096,
            virtual_batch_size=1024,
            num_workers=0,
            drop_last=False,
            callbacks=[TQDMCallback(total_epochs=200)]
        )

    def _create_tabnet_model(self):
        return TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-3),
            scheduler_params={"step_size":50, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=0
        )

    def predict(self, X):
        entry_probs = self.entry_model.predict_proba(X)[:, 1]
        direction_probs = self.direction_model.predict_proba(X)[:, 1]
        return entry_probs, direction_probs

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)

        # ✅ 저장 전 callback 제거
        if hasattr(self.entry_model, '_callback_container'):
            self.entry_model._callback_container = None
        if hasattr(self.direction_model, '_callback_container'):
            self.direction_model._callback_container = None

        joblib.dump(self.entry_model, f"{self.model_dir}/tabnet_entry_model.pkl")
        joblib.dump(self.direction_model, f"{self.model_dir}/tabnet_direction_model.pkl")

    def load(self):
        self.entry_model = joblib.load(f"{self.model_dir}/tabnet_entry_model.pkl")
        self.direction_model = joblib.load(f"{self.model_dir}/tabnet_direction_model.pkl")

# ✅ Usage Example
if __name__ == "__main__":
    np.random.seed(42)

    DATA_DIR = "data/dualbranch/"

    X_minute = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_X_minute_full.npy"))
    X_daily = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_X_daily_full.npy"))
    y_entry = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_y_entry_full.npy"))
    y_direction = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_y_direction_full.npy"))

    # Flatten features
    X_minute_last = X_minute[:, -1, :]
    X_daily_last = X_daily[:, -1, :]
    X_features = np.concatenate([X_minute_last, X_daily_last], axis=1).astype(np.float32)

    trainer = TabNetTrainer(feature_dim=X_features.shape[1])

    os.makedirs("testlog", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"testlog/train_tabnet_models_log_{timestamp}.txt"

    with open(log_path, "w") as f:
        trainer.train(X_features, y_entry, y_direction)

        entry_probs, direction_probs = trainer.predict(X_features)

        for i in tqdm(range(min(10, len(entry_probs))), desc="Logging Predictions"):
            log_text = f"Entry Prob: {entry_probs[i]:.4f}, Direction Prob: {direction_probs[i]:.4f}\n"
            print(log_text.strip())
            f.write(log_text)

    trainer.save()
    trainer.load()

    print("All process completed.")
