
import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/processed/sol1m_with_trend.csv"
OUTPUT_DIR = "data/multitask/"
MINUTE_SEQ_LEN = 30
DAILY_SEQ_LEN = 10
HORIZON = 1
THRESHOLD = 0.0005
MIN_OBS = 1e-5

def make_debug_dataset():
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
            print(f"✅ 첫 시퀀스 생성 성공! 날짜: {sample_date}")
            break  # 하나만 생성하고 중단
        except Exception as e:
            print(f"⛔ 실패 at {sample_date} → {e}")
            continue

    if X_minute:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(os.path.join(OUTPUT_DIR, "debug_X_minute.npy"), np.array(X_minute, dtype=np.float32))
        np.save(os.path.join(OUTPUT_DIR, "debug_X_daily.npy"), np.array(X_daily, dtype=np.float32))
        np.save(os.path.join(OUTPUT_DIR, "debug_y_entry.npy"), np.array(y_entry, dtype=np.int64))
        np.save(os.path.join(OUTPUT_DIR, "debug_y_direction.npy"), np.array(y_direction, dtype=np.int64))
        print("✅ 디버깅용 샘플 저장 완료")
    else:
        print("❌ 여전히 시퀀스 생성 실패")

if __name__ == "__main__":
    make_debug_dataset()
