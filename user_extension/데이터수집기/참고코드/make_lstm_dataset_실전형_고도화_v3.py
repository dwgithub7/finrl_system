
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# ✅ 설정
INPUT_PATH = "data/processed/SOLUSDT_1m_with_trend.csv"
OUTPUT_X_PATH = "data/dataset/SOLUSDT_X_lstm.npy"
OUTPUT_Y_PATH = "data/dataset/SOLUSDT_y_lstm.npy"
OUTPUT_META_PATH = "data/dataset/SOLUSDT_meta.json"
TASK_TYPE = "classification"
WINDOW_SIZE = 60
HORIZON = 1
INITIAL_THRESHOLD = 0.001
MIN_THRESHOLD = 0.0003
MIN_CLASS_RATIO = 0.05  # 최소 클래스 비율 (ex: 관망 비중 최소 5%)

def classify_return(x, threshold):
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0

def adjust_threshold(df, threshold, min_threshold, min_ratio):
    t = threshold
    while True:
        y_temp = df["future_return"].apply(lambda x: classify_return(x, t))
        class_ratio = y_temp.value_counts(normalize=True)
        if class_ratio.get(0, 0) < min_ratio and t > min_threshold:
            t *= 0.8
        else:
            break
    return t

def rebalance_multiclass(X, y):
    from collections import Counter
    print("⚖️ 클래스 조정 전:", dict(Counter(y)))

    X_flat = X.reshape((X.shape[0], -1))
    rus = RandomUnderSampler()
    ros = RandomOverSampler()
    X_rus, y_rus = rus.fit_resample(X_flat, y)
    X_ros, y_ros = ros.fit_resample(X_rus, y_rus)
    X_final = X_ros.reshape((-1, X.shape[1], X.shape[2]))

    print("⚖️ 클래스 조정 후:", dict(Counter(y_ros)))
    return X_final, y_ros

def make_lstm_dataset():
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df["future_return"] = df["log_return"].shift(-HORIZON)
    df = df[df["future_return"].abs() > 1e-5].dropna()

    # 1️⃣ threshold 조정
    threshold = adjust_threshold(df, INITIAL_THRESHOLD, MIN_THRESHOLD, MIN_CLASS_RATIO)
    df["target"] = df["future_return"].apply(lambda x: classify_return(x, threshold))
    print(f"⚙️ 최종 threshold: {threshold:.6f}")

    # 2️⃣ 피처 자동 감지
    exclude_cols = ["future_return", "target"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # 3️⃣ 슬라이딩 윈도우 생성
    X, y = [], []
    for i in range(WINDOW_SIZE, len(df) - HORIZON):
        X.append(df.iloc[i - WINDOW_SIZE:i][feature_cols].values)
        y.append(df.iloc[i]["target"])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # 4️⃣ 삼분 클래스 균형화
    X, y = rebalance_multiclass(X, y)

    # 5️⃣ 클래스 가중치
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    print("📌 클래스 가중치:", dict(zip(np.unique(y), class_weights.tolist())))

    # 📊 시각화
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts)
    plt.title("Rebalanced Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    # 💾 저장
    os.makedirs(os.path.dirname(OUTPUT_X_PATH), exist_ok=True)
    np.save(OUTPUT_X_PATH, X)
    np.save(OUTPUT_Y_PATH, y)
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": threshold,
            "class_weights": class_weights.tolist(),
            "feature_cols": feature_cols,
            "X_shape": X.shape,
            "y_shape": y.shape,
        }, f, indent=2, ensure_ascii=False)

    print("💾 저장 완료:", OUTPUT_X_PATH, OUTPUT_Y_PATH, OUTPUT_META_PATH)
    return X, y

# ✅ 실행
if __name__ == "__main__":
    make_lstm_dataset()
