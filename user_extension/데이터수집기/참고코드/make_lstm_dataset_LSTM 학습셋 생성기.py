
import pandas as pd
import numpy as np
import os

# ✅ 설정값
INPUT_PATH = "data/processed/SOLUSDT_1m_with_trend.csv"
WINDOW_SIZE = 60           # 60분 입력
HORIZON = 1                # 1분 뒤 예측
THRESHOLD = 0.001          # 타겟 분류 임계값 (±0.1%)

FEATURE_COLS = [
    "close", "volume", "return", "log_return",
    "ma_5", "ma_20", "ma_diff", "volatility_5",
    "bollinger_width", "momentum_3", "momentum_10",
    "price_to_ma_5", "volume_ratio_1m", "volume_ma_5",
    "body_size", "shadow_ratio", "daily_trend"
]

def classify_return(x, threshold=THRESHOLD):
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0

def make_lstm_dataset(input_csv=INPUT_PATH, window_size=WINDOW_SIZE, horizon=HORIZON):
    df = pd.read_csv(input_csv, index_col=0, parse_dates=True)

    # 타겟 생성
    df["future_return"] = df["log_return"].shift(-horizon)
    df["target_class"] = df["future_return"].apply(classify_return)

    df = df.dropna()

    X, y = [], []

    for i in range(window_size, len(df) - horizon):
        X.append(df.iloc[i-window_size:i][FEATURE_COLS].values)
        y.append(df.iloc[i]["target_class"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"✅ LSTM 데이터셋 생성 완료: X.shape = {X.shape}, y.shape = {y.shape}")
    print("🔹 샘플 X[0]:\n", X[0])
    print("🔹 샘플 y[0]:", y[0])
    return X, y

# ✅ 실행
if __name__ == "__main__":
    make_lstm_dataset()
