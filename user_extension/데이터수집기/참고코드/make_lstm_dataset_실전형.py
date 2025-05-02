
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from joblib import dump

# ✅ 설정
INPUT_PATH = "data/processed/SOLUSDT_1m_with_trend.csv"
OUTPUT_X_PATH = "data/dataset/SOLUSDT_X_lstm.npy"
OUTPUT_Y_PATH = "data/dataset/SOLUSDT_y_lstm.npy"
TASK_TYPE = "classification"  # 또는 "regression"
WINDOW_SIZE = 60
HORIZON = 1
THRESHOLD = 0.001

def classify_return(x, threshold):
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0

def make_lstm_dataset(input_csv=INPUT_PATH, window_size=WINDOW_SIZE, horizon=HORIZON, 
                      threshold=THRESHOLD, task=TASK_TYPE):
    df = pd.read_csv(input_csv, index_col=0, parse_dates=True)

    df["future_return"] = df["log_return"].shift(-horizon)

    if task == "classification":
        df["target"] = df["future_return"].apply(lambda x: classify_return(x, threshold))
    elif task == "regression":
        df["target"] = df["future_return"]
    else:
        raise ValueError("TASK_TYPE must be 'classification' or 'regression'")

    df = df.dropna()

    # 🔍 피처 자동 감지
    exclude_cols = ["future_return", "target"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"🧩 사용 피처 ({len(feature_cols)}개):", feature_cols)

    # 🔁 슬라이딩 윈도우 생성
    X, y = [], []
    for i in range(window_size, len(df) - horizon):
        X.append(df.iloc[i - window_size:i][feature_cols].values)
        y.append(df.iloc[i]["target"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32 if task == "regression" else np.int64)

    print(f"✅ 데이터셋 생성 완료: X.shape = {X.shape}, y.shape = {y.shape}")

    # 💾 저장
    os.makedirs(os.path.dirname(OUTPUT_X_PATH), exist_ok=True)
    np.save(OUTPUT_X_PATH, X)
    np.save(OUTPUT_Y_PATH, y)
    print(f"💾 저장 완료: {OUTPUT_X_PATH}, {OUTPUT_Y_PATH}")

    # 📊 분포 시각화
    if task == "classification":
        unique, counts = np.unique(y, return_counts=True)
        print("📊 클래스 분포:", dict(zip(unique, counts)))
        plt.bar(unique, counts)
        plt.title("Target Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

    return X, y

# ✅ 실행
if __name__ == "__main__":
    make_lstm_dataset()
