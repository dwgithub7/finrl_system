
import pandas as pd
import numpy as np
import os

# ✅ 설정
INPUT_PATH = "data/processed/sol1m_with_trend.csv"
OUTPUT_DIR = "data/multitask/"
MINUTE_SEQ_LEN = 30
DAILY_SEQ_LEN = 10
HORIZON = 1
THRESHOLD = 0.0005
MIN_OBS = 1e-5

def make_dataset_fast():
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["future_return"] = df["log_return"].shift(-HORIZON)
    df = df[df["future_return"].abs() > MIN_OBS].dropna()
    df["y_entry"] = (df["future_return"].abs() > THRESHOLD).astype(int)
    df["y_direction"] = (df["future_return"] > 0).astype(int)
    df["date"] = df.index.date

    df_daily = df.groupby("date").mean().dropna()
    minute_cols = [c for c in df.columns if c not in ["future_return", "y_entry", "y_direction", "date"]]
    daily_cols = [c for c in df_daily.columns if c not in ["future_return", "y_entry", "y_direction"]]

    df_min = df[minute_cols + ["y_entry", "y_direction", "date"]].copy().reset_index()
    df_np = df_min[minute_cols].values
    label_entry = df_min["y_entry"].values
    label_dir = df_min["y_direction"].values
    date_list = df_min["date"].values

    X_minute, X_daily, y_entry, y_direction = [], [], [], []

    daily_index_lookup = {date: idx for idx, date in enumerate(df_daily.index)}

    for i in range(MINUTE_SEQ_LEN, len(df_np) - HORIZON):
        sample_date = date_list[i]
        daily_idx = daily_index_lookup.get(sample_date, -1)
        if daily_idx < DAILY_SEQ_LEN:
            continue
        try:
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
    make_dataset_fast()
