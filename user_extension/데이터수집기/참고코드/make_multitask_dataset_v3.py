
import pandas as pd
import numpy as np
import os

# ✅ 설정값
INPUT_PATH = "data/processed/SOLUSDT_1m_with_trend.csv"
OUTPUT_DIR = "data/multitask/"
WINDOW_SIZE = 60
HORIZON = 1
INITIAL_THRESHOLD = 0.0005
MIN_OBS = 1e-5
MAX_ENTRY_RATIO = 0.95  # 진입 비중이 95% 넘으면 threshold ↑ 조정

X_PATH = os.path.join(OUTPUT_DIR, "SOLUSDT_X.npy")
Y_ENTRY_PATH = os.path.join(OUTPUT_DIR, "SOLUSDT_y_entry.npy")
Y_DIRECTION_PATH = os.path.join(OUTPUT_DIR, "SOLUSDT_y_direction.npy")

def make_multitask_dataset():
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df["future_return"] = df["log_return"].shift(-HORIZON)
    df = df[df["future_return"].abs() > MIN_OBS].dropna()

    threshold = INITIAL_THRESHOLD
    while True:
        df["y_entry"] = (df["future_return"].abs() > threshold).astype(int)
        entry_ratio = df["y_entry"].mean()
        if entry_ratio <= MAX_ENTRY_RATIO or threshold >= 0.01:
            break
        threshold *= 1.2

    df["y_direction"] = df["future_return"].apply(lambda x: 1 if x > 0 else 0)

    # 피처 선택
    exclude_cols = ["future_return", "y_entry", "y_direction"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X, y_entry, y_direction = [], [], []
    for i in range(WINDOW_SIZE, len(df) - HORIZON):
        X.append(df.iloc[i - WINDOW_SIZE:i][feature_cols].values)
        y_entry.append(df.iloc[i]["y_entry"])
        y_direction.append(df.iloc[i]["y_direction"])

    X = np.array(X, dtype=np.float32)
    y_entry = np.array(y_entry, dtype=np.int64)
    y_direction = np.array(y_direction, dtype=np.int64)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(X_PATH, X)
    np.save(Y_ENTRY_PATH, y_entry)
    np.save(Y_DIRECTION_PATH, y_direction)

    print(f"✅ threshold 조정 완료: {threshold:.6f} (entry ratio: {entry_ratio:.2%})")
    print(f"✅ 저장 완료: {X_PATH}, {Y_ENTRY_PATH}, {Y_DIRECTION_PATH}")
    print(f"📐 X.shape={X.shape}, y_entry.shape={y_entry.shape}, y_direction.shape={y_direction.shape}")
    return X, y_entry, y_direction

if __name__ == "__main__":
    make_multitask_dataset()
