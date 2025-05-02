
import pandas as pd
import numpy as np
import os

# ✅ 설정값
INPUT_PATH = "data/processed/sol1m_with_trend.csv"
OUTPUT_DIR = "data/multitask/"
MINUTE_SEQ_LEN = 60
DAILY_SEQ_LEN = 20
HORIZON = 1
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
    df["trend_slope"] = df["ma_5"] - df["ma_5"].shift(1)
    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["signal"] = df["macd"].ewm(span=9).mean()
    df["macd_diff"] = df["macd"] - df["signal"]
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff().abs()
    tr_smooth = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / (tr_smooth + 1e-8))
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / (tr_smooth + 1e-8))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)) * 100
    df["adx"] = dx.rolling(window=14).mean()
    return df

def extract_best_test_window():
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df = add_features(df)
    df["future_return"] = df["log_return"].shift(-HORIZON)
    df = df[df["future_return"].abs() > MIN_OBS].dropna()
    df["y_entry"] = (df["future_return"].abs() > 0.0005).astype(int)
    df["y_direction"] = df["future_return"].apply(lambda x: 1 if x > 0 else 0)
    df["date"] = df.index.date
    df_daily = df.groupby("date").mean()
    dates = sorted(df["date"].unique(), reverse=True)

    for target_date in dates:
        sub_df = df[df["date"] == target_date]
        if len(sub_df) < MINUTE_SEQ_LEN:
            continue
        try:
            daily_index = df_daily.index.get_loc(target_date)
            if daily_index < DAILY_SEQ_LEN:
                continue
            minute_features = [col for col in df.columns if col not in ["future_return", "y_entry", "y_direction", "date"]]
            daily_features = [col for col in df_daily.columns if col not in ["future_return", "y_entry", "y_direction"]]
            X_minute, X_daily, y_entry, y_direction = [], [], [], []

            for i in range(sub_df.index[0], sub_df.index[-1]):
                if i - MINUTE_SEQ_LEN < 0 or i + HORIZON >= len(df):
                    continue
                if df.index[i].date() != target_date:
                    continue
                minute_seq = df.iloc[i - MINUTE_SEQ_LEN:i][minute_features].values
                daily_seq = df_daily.iloc[daily_index - DAILY_SEQ_LEN:daily_index][daily_features].values
                if len(minute_seq) == MINUTE_SEQ_LEN and len(daily_seq) == DAILY_SEQ_LEN:
                    X_minute.append(minute_seq)
                    X_daily.append(daily_seq)
                    y_entry.append(df.iloc[i]["y_entry"])
                    y_direction.append(df.iloc[i]["y_direction"])
            if len(X_minute) > 0:
                print(f"✅ 추출된 테스트 일자: {target_date} ({len(X_minute)}개 샘플)")
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                np.save(OUTPUT_DIR + "realtest_X_minute.npy", np.array(X_minute, dtype=np.float32))
                np.save(OUTPUT_DIR + "realtest_X_daily.npy", np.array(X_daily, dtype=np.float32))
                np.save(OUTPUT_DIR + "realtest_y_entry.npy", np.array(y_entry, dtype=np.int64))
                np.save(OUTPUT_DIR + "realtest_y_direction.npy", np.array(y_direction, dtype=np.int64))
                return
        except:
            continue
    print("❌ 조건 만족하는 테스트 구간을 찾지 못했습니다.")

if __name__ == "__main__":
    extract_best_test_window()
