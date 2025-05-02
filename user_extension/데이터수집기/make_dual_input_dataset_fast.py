
import pandas as pd
import numpy as np
import os

# ✅ 설정값
INPUT_PATH = "data/processed/sol1m_with_trend.csv"
OUTPUT_DIR = "data/multitask/"
MINUTE_SEQ_LEN = 30
DAILY_SEQ_LEN = 10
HORIZON = 1
THRESHOLD = 0.0005
MAX_ENTRY_RATIO = 0.95
MIN_OBS = 1e-5

def add_features(df):
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"] - 1
    df["volatility_5"] = df["log_return"].rolling(window=5).std()
    df["volume_change"] = df["volume"].pct_change()
    df["candle_body"] = (df["close"] - df["open"]).abs()
    df["candle_range"] = df["high"] - df["low"]
    df["body_ratio"] = df["candle_body"] / df["candle_range"].replace(0, np.nan)
    df["trend_diff"] = df["ma_5"] - df["ma_20"]
    df["trend_slope"] = df["ma_5"].diff()

    # MACD, Signal, Diff
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9).mean()
    df["macd_diff"] = df["macd"] - df["signal"]

    # RSI
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))

    # OBV 벡터화
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (df["volume"] * direction).fillna(0).cumsum()

    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()

    # ADX
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = df["low"].diff().abs()
    tr_smooth = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / (tr_smooth + 1e-8))
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / (tr_smooth + 1e-8))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)) * 100
    df["adx"] = dx.rolling(window=14).mean()

    return df

def make_dual_dataset():
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df = add_features(df)
    df["future_return"] = df["log_return"].shift(-HORIZON)
    df = df[df["future_return"].abs() > MIN_OBS].dropna()


    # threshold 자동조절
    # threshold = THRESHOLD
    # while True:
    #     df["y_entry"] = (df["future_return"].abs() > threshold).astype(int)
    #     if df["y_entry"].mean() <= MAX_ENTRY_RATIO or threshold >= 0.01:
    #         break
    #     threshold *= 1.2
    df["y_entry"] = (df["future_return"].abs() > THRESHOLD).astype(int)


    df["y_direction"] = (df["future_return"] > 0).astype(int)
    df["date"] = df.index.date
    df_daily = df.groupby("date").mean().dropna()

    minute_cols = [c for c in df.columns if c not in ["future_return", "y_entry", "y_direction", "date"]]
    daily_cols = [c for c in df_daily.columns if c not in ["future_return", "y_entry", "y_direction"]]

    df_min = df[minute_cols + ["y_entry", "y_direction", "date"]].copy().reset_index()
    df_daily = df_daily[daily_cols]

    X_minute, X_daily, y_entry, y_direction = [], [], [], []

    df_np = df_min[minute_cols].values
    label_entry = df_min["y_entry"].values
    label_dir = df_min["y_direction"].values
    date_list = df_min["date"].values

    for i in range(MINUTE_SEQ_LEN, len(df_np) - HORIZON):
        sample_date = date_list[i]
        try:
            daily_idx = df_daily.index.get_loc(sample_date)
            if daily_idx < DAILY_SEQ_LEN:
                continue
            min_seq = df_np[i - MINUTE_SEQ_LEN:i]
            daily_seq = df_daily.iloc[daily_idx - DAILY_SEQ_LEN:daily_idx].values
            X_minute.append(min_seq)
            X_daily.append(daily_seq)
            y_entry.append(label_entry[i])
            y_direction.append(label_dir[i])
        except:
            continue

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_minute.npy"), np.array(X_minute, dtype=np.float32))
    np.save(os.path.join(OUTPUT_DIR, "X_daily.npy"), np.array(X_daily, dtype=np.float32))
    np.save(os.path.join(OUTPUT_DIR, "y_entry.npy"), np.array(y_entry, dtype=np.int64))
    np.save(os.path.join(OUTPUT_DIR, "y_direction.npy"), np.array(y_direction, dtype=np.int64))

    print(f"✅ 저장 완료: {len(X_minute)}개 샘플")
    print(f"📐 X_minute.shape = {np.array(X_minute).shape}")
    print(f"📐 X_daily.shape = {np.array(X_daily).shape}")

if __name__ == "__main__":
    make_dual_dataset()
