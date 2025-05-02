
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ✅ 설정
INPUT_PATH = "data/processed/SOLUSDT_1m_with_trend.csv"
OUTPUT_X_PATH = "data/dataset/SOLUSDT_X_lstm.npy"
OUTPUT_Y_PATH = "data/dataset/SOLUSDT_y_lstm.npy"
OUTPUT_META_PATH = "data/dataset/SOLUSDT_meta.json"
TASK_TYPE = "classification"
WINDOW_SIZE = 60
HORIZON = 1
THRESHOLD = 0.001  # 초기값

def classify_return(x, threshold):
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0

def rebalance_dataset(X, y, method="both"):
    from collections import Counter
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    print("⚖️ 클래스 재조정 전:", dict(Counter(y)))
    X_flat = X.reshape((X.shape[0], -1))

    if method == "under":
        rus = RandomUnderSampler()
        X_res, y_res = rus.fit_resample(X_flat, y)
    elif method == "over":
        ros = RandomOverSampler()
        X_res, y_res = ros.fit_resample(X_flat, y)
    elif method == "both":
        rus = RandomUnderSampler()
        ros = RandomOverSampler()
        X_temp, y_temp = rus.fit_resample(X_flat, y)
        X_res, y_res = ros.fit_resample(X_temp, y_temp)
    else:
        return X, y

    X_res = X_res.reshape((-1, X.shape[1], X.shape[2]))
    print("⚖️ 클래스 재조정 후:", dict(Counter(y_res)))
    return X_res, y_res

def make_lstm_dataset(input_csv=INPUT_PATH, window_size=WINDOW_SIZE, horizon=HORIZON,
                      threshold=THRESHOLD, task=TASK_TYPE):

    df = pd.read_csv(input_csv, index_col=0, parse_dates=True)
    df["future_return"] = df["log_return"].shift(-horizon)
    df = df[df["future_return"].abs() > 1e-5]
    df.dropna(inplace=True)

    # 1️⃣ 관망 비율 보정 (threshold 조정)
    def adjust_threshold(df, init_threshold):
        t = init_threshold
        while True:
            y_temp = df["future_return"].apply(lambda x: classify_return(x, t))
            counts = y_temp.value_counts(normalize=True)
            if counts.get(0, 0) < 0.05:
                t *= 0.8  # 더 민감하게
            else:
                break
            if t < 0.0001:
                break
        return t

    adjusted_threshold = adjust_threshold(df, threshold)
    print(f"⚙️ 사용된 threshold: {adjusted_threshold:.6f}")
    df["target"] = df["future_return"].apply(lambda x: classify_return(x, adjusted_threshold))

    # 피처 자동 인식
    exclude_cols = ["future_return", "target"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # 윈도우 기반 시계열 생성
    X, y = [], []
    for i in range(window_size, len(df) - horizon):
        X.append(df.iloc[i - window_size:i][feature_cols].values)
        y.append(df.iloc[i]["target"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # 2️⃣ + 3️⃣ 언더샘플링 + 오버샘플링 병행
    X, y = rebalance_dataset(X, y, method="both")

    print(f"✅ 재조정 후: X.shape = {X.shape}, y.shape = {y.shape}")

    # 클래스 가중치
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    print("📌 클래스 가중치:", dict(zip(np.unique(y), class_weights.tolist())))

    # 시각화
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts)
    plt.title("Target Class Distribution (Rebalanced)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    # 저장
    os.makedirs(os.path.dirname(OUTPUT_X_PATH), exist_ok=True)
    np.save(OUTPUT_X_PATH, X)
    np.save(OUTPUT_Y_PATH, y)

    meta = {
        "threshold": adjusted_threshold,
        "feature_cols": feature_cols,
        "X_shape": X.shape,
        "y_shape": y.shape,
        "class_weights": class_weights.tolist()
    }
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("💾 저장 완료:", OUTPUT_X_PATH, OUTPUT_Y_PATH, OUTPUT_META_PATH)
    return X, y

# ✅ 실행
if __name__ == "__main__":
    make_lstm_dataset()
