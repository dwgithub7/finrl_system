
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# ✅ 사용자 입력 설정 (이 부분만 바꾸면 됨)
MINUTE_CSV_PATH = "data/raw/SOLUSDT_future_1m_202501010000~202504222359.csv"
DAILY_CSV_PATH = "data/raw/resampled/SOLUSDT_future_1m_202501010000~202504222359_1d.csv"
OUTPUT_PATH = "data/processed/sol1m_with_trend.csv"

def classify_daily_trend(df_daily):
    ma5 = df_daily["close"].rolling(5).mean()
    ma20 = df_daily["close"].rolling(20).mean()
    return np.where(ma5 > ma20, 1, -1)

def generate_features(df_1m, df_daily):
    df_daily["trend_label"] = classify_daily_trend(df_daily)
    df_daily = df_daily[["trend_label"]].copy()
    df_daily.index = pd.to_datetime(df_daily.index)
    df_1m["daily_trend"] = df_daily.reindex(df_1m.index, method="ffill")["trend_label"]

    df_1m["return"] = df_1m["close"].pct_change()
    df_1m["log_return"] = np.log(df_1m["close"] / df_1m["close"].shift(1))
    df_1m["ma_5"] = df_1m["close"].rolling(5).mean()
    df_1m["ma_20"] = df_1m["close"].rolling(20).mean()
    df_1m["ma_diff"] = df_1m["ma_5"] - df_1m["ma_20"]
    df_1m["volatility_5"] = df_1m["log_return"].rolling(5).std() * np.sqrt(5)
    df_1m["bollinger_width"] = (df_1m["ma_5"].rolling(5).max() - df_1m["ma_5"].rolling(5).min()) / df_1m["ma_5"]
    df_1m["momentum_3"] = df_1m["close"] - df_1m["close"].shift(3)
    df_1m["momentum_10"] = df_1m["close"] - df_1m["close"].shift(10)
    df_1m["price_to_ma_5"] = (df_1m["close"] - df_1m["ma_5"]) / df_1m["ma_5"]
    df_1m["volume_ratio_1m"] = df_1m["volume"] / df_1m["volume"].shift(1)
    df_1m["volume_ma_5"] = df_1m["volume"].rolling(5).mean()
    df_1m["body_size"] = df_1m["close"] - df_1m["open"]
    df_1m["shadow_ratio"] = (df_1m["high"] - df_1m["low"]) / (abs(df_1m["body_size"]) + 1e-9)

    df_1m = df_1m.dropna()
    return df_1m

def preprocess_pipeline(minute_csv_path, daily_csv_path, output_path):
    df_1m = pd.read_csv(minute_csv_path, index_col=0, parse_dates=True)
    df_daily = pd.read_csv(daily_csv_path, index_col=0, parse_dates=True)

    df = generate_features(df_1m, df_daily)

    feature_cols = [
        "close", "volume", "return", "log_return",
        "ma_5", "ma_20", "ma_diff", "volatility_5",
        "bollinger_width", "momentum_3", "momentum_10",
        "price_to_ma_5", "volume_ratio_1m", "volume_ma_5",
        "body_size", "shadow_ratio", "daily_trend"
    ]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = df_scaled[feature_cols].replace([np.inf, -np.inf], np.nan)
    df_scaled = df_scaled.dropna(subset=feature_cols)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scaled.to_csv(output_path)
    print(f"✅ 전처리 완료: {output_path}")

    return df_scaled

# ✅ 바로 실행
if __name__ == "__main__":
    preprocess_pipeline(MINUTE_CSV_PATH, DAILY_CSV_PATH, OUTPUT_PATH)
