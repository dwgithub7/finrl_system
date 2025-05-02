import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import early_stopping

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
        os.makedirs(self.model_dir, exist_ok=True)  # ✅ 디렉토리 없으면 생성
        joblib.dump(self.entry_model, f"{self.model_dir}/lightgbm_entry_model.pkl")
        joblib.dump(self.direction_model, f"{self.model_dir}/lightgbm_direction_model.pkl")

    def load(self):
        self.entry_model = joblib.load(f"{self.model_dir}/lightgbm_entry_model.pkl")
        self.direction_model = joblib.load(f"{self.model_dir}/lightgbm_direction_model.pkl")

# ✅ Usage Example (Dummy Data)
if __name__ == "__main__":
    np.random.seed(42)

    feature_dim = 50
    num_samples = 1000

    X = np.random.randn(num_samples, feature_dim)
    entry_y = np.random.randint(0, 2, size=num_samples)
    direction_y = np.random.randint(0, 2, size=num_samples)

    trainer = LightGBMTrainer(feature_dim=feature_dim)
    trainer.train(X, entry_y, direction_y)
    entry_probs, direction_probs = trainer.predict(X)

    print(f"Entry Probabilities (first 5): {entry_probs[:5]}")
    print(f"Direction Probabilities (first 5): {direction_probs[:5]}")

    trainer.save()
    trainer.load()
