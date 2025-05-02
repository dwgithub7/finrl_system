import lightgbm as lgb
import pandas as np
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import early_stopping
from datetime import datetime

# ✅ LightGBM Trainer Class
class LightGBMTrainer:
    def __init__(self, feature_dim, model_dir="models"):
        self.feature_dim = feature_dim
        self.model_dir = model_dir
        self.entry_model = None
        self.direction_model = None

    def train(self, X, entry_y, direction_y):
        # Split data
        X_train, X_val, entry_y_train, entry_y_val = train_test_split(X, entry_y, test_size=0.2, random_state=42)
        _, _, direction_y_train, direction_y_val = train_test_split(X, direction_y, test_size=0.2, random_state=42)

        # Train Entry model
        entry_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1
        }
        entry_train = lgb.Dataset(X_train, label=entry_y_train)
        entry_valid = lgb.Dataset(X_val, label=entry_y_val)

        self.entry_model = lgb.train(
            entry_params,
            entry_train,
            valid_sets=[entry_valid],
            num_boost_round=500,
            callbacks=[early_stopping(stopping_rounds=30)]
        )

        # Train Direction model
        direction_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1
        }
        direction_train = lgb.Dataset(X_train, label=direction_y_train)
        direction_valid = lgb.Dataset(X_val, label=direction_y_val)

        self.direction_model = lgb.train(
            direction_params,
            direction_train,
            valid_sets=[direction_valid],
            num_boost_round=500,
            callbacks=[early_stopping(stopping_rounds=30)]
        )

    def predict(self, X):
        entry_probs = self.entry_model.predict(X)
        direction_probs = self.direction_model.predict(X)
        return entry_probs, direction_probs

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.entry_model, f"{self.model_dir}/lightgbm_entry_model.pkl")
        joblib.dump(self.direction_model, f"{self.model_dir}/lightgbm_direction_model.pkl")

    def load(self):
        self.entry_model = joblib.load(f"{self.model_dir}/lightgbm_entry_model.pkl")
        self.direction_model = joblib.load(f"{self.model_dir}/lightgbm_direction_model.pkl")

# ✅ Usage Example (Load from SOLUSDT npy Files)
if __name__ == "__main__":
    np.random.seed(42)

    DATA_DIR = "data/dualbranch/"

    X_minute = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_X_minute_full.npy"))
    X_daily = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_X_daily_full.npy"))
    y_entry = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_y_entry_full.npy"))
    y_direction = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_y_direction_full.npy"))

    # Flatten features: Use last timestep
    X_minute_last = X_minute[:, -1, :]  # (batch, feature_dim_minute)
    X_daily_last = X_daily[:, -1, :]    # (batch, feature_dim_daily)

    # Concatenate minute and daily features
    X_features = np.concatenate([X_minute_last, X_daily_last], axis=1)  # (batch, total_feature_dim)

    trainer = LightGBMTrainer(feature_dim=X_features.shape[1])

    os.makedirs("testlog", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"testlog/train_lightgbm_models_log_{timestamp}.txt"

    with open(log_path, "w") as f:
        trainer.train(X_features, y_entry, y_direction)
        entry_probs, direction_probs = trainer.predict(X_features)

        for i in range(min(10, len(entry_probs))):
            log_text = f"Entry Prob: {entry_probs[i]:.4f}, Direction Prob: {direction_probs[i]:.4f}\n"
            print(log_text.strip())
            f.write(log_text)

    trainer.save()
    trainer.load()
