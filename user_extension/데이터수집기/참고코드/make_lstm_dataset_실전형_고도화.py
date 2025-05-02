
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
TASK_TYPE = "classification"  # or "regression"
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

    # 🔻 약한 변동 샘플 제거 (노이즈 감소)
    df = df[df["future_return"].abs() > 1e-5]

    if task == "classification":
        df["target"] = df["future_return"].apply(lambda x: classify_return(x, threshold))
    elif task == "regression":
        df["target"] = df["future_return"]
    else:
        raise ValueError("TASK_TYPE must be 'classification' or 'regression'")

    df = df.dropna()

    # 🔍 피처 자동 인식
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

    # 📊 클래스 분포 확인
    class_weights = None
    if task == "classification":
        unique, counts = np.unique(y, return_counts=True)
        print("📊 클래스 분포:", dict(zip(unique, counts)))
        plt.bar(unique, counts)
        plt.title("Target Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

        class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
        print("📌 클래스 가중치:", dict(zip(np.unique(y), class_weights.tolist())))

    # 💾 저장
    os.makedirs(os.path.dirname(OUTPUT_X_PATH), exist_ok=True)
    np.save(OUTPUT_X_PATH, X)
    np.save(OUTPUT_Y_PATH, y)

    # 💾 메타데이터 저장
    meta = {
        "input_path": input_csv,
        "X_shape": X.shape,
        "y_shape": y.shape,
        "window_size": window_size,
        "horizon": horizon,
        "task": task,
        "threshold": threshold,
        "feature_cols": feature_cols,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "timestamp_range": [str(df.index[0]), str(df.index[-1])]
    }

    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"💾 저장 완료: {OUTPUT_X_PATH}, {OUTPUT_Y_PATH}, {OUTPUT_META_PATH}")
    return X, y

# ✅ 실행
if __name__ == "__main__":
    make_lstm_dataset()
