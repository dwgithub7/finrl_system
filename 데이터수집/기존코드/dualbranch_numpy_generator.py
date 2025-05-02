# dualbranch_numpy_generator_fast.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 🔥 설정
input_minute_file = "data/processed/SOLUSDT_1m_finrl_processed_20240101~20250426.csv"
save_dir = "data/numpy"
os.makedirs(save_dir, exist_ok=True)

filename = os.path.basename(input_minute_file).replace(".csv", "")
parts = filename.split("_")
symbol_tf = f"{parts[0]}_{parts[1]}"
period = parts[-1]

X_minute_file = f"{symbol_tf}_X_minute_{period}.npy"
X_daily_file = f"{symbol_tf}_X_daily_{period}.npy"
y_entry_file = f"{symbol_tf}_y_entry_{period}.npy"
y_direction_file = f"{symbol_tf}_y_direction_{period}.npy"

# 📥 데이터 로딩
df_minute = pd.read_csv(input_minute_file)
df_minute['date'] = pd.to_datetime(df_minute['date'])
df_minute = df_minute.sort_values('date')

# 📥 일봉 생성
df_daily = df_minute.resample('1D', on='date').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
    'macd': 'last', 'rsi': 'last', 'cci': 'last', 'adx': 'last', 'atr': 'last',
    'obv': 'last', 'ma5': 'last', 'ma10': 'last', 'ma20': 'last',
    'body_size': 'last', 'shadow_ratio': 'last'
}).dropna().reset_index()

# ✅ 라벨 생성
df_minute['future_return'] = df_minute['close'].shift(-1) / df_minute['close'] - 1
threshold = 0.0002
df_minute['entry_label'] = (df_minute['future_return'].abs() > threshold).astype(int)
df_minute['direction_label'] = (df_minute['future_return'] > 0).astype(int)

# 🔵 numpy 변환
minute_array = df_minute[[
    "open", "high", "low", "close", "volume",
    "macd", "rsi", "cci", "adx", "atr",
    "obv", "ma5", "ma10", "ma20", "body_size", "shadow_ratio"
]].values

daily_array = df_daily[[
    "open", "high", "low", "close", "volume",
    "macd", "rsi", "cci", "adx", "atr",
    "obv", "ma5", "ma10", "ma20", "body_size", "shadow_ratio"
]].values

date_array_minute = df_minute['date'].values
date_array_daily = df_daily['date'].values

entry_labels = df_minute['entry_label'].values
direction_labels = df_minute['direction_label'].values

# 🔵 파라미터
WINDOW_SIZE_MIN = 30
WINDOW_SIZE_DAY = 10

# 🔵 결과 저장용
X_minute = []
X_daily = []
y_entry = []
y_direction = []

print("✅ Numpy 변환 시작...")
for idx in tqdm(range(max(WINDOW_SIZE_MIN, WINDOW_SIZE_DAY), len(minute_array)-1)):
    end_time = date_array_minute[idx]
    
    # 분봉 윈도우
    minute_window = minute_array[idx-WINDOW_SIZE_MIN:idx]
    
    # 일봉 윈도우
    available_days = date_array_daily[date_array_daily < end_time]
    if len(available_days) < WINDOW_SIZE_DAY:
        continue
    latest_days_idx = np.where(date_array_daily < end_time)[0][-WINDOW_SIZE_DAY:]
    daily_window = daily_array[latest_days_idx]
    
    if minute_window.shape[0] == WINDOW_SIZE_MIN and daily_window.shape[0] == WINDOW_SIZE_DAY:
        X_minute.append(minute_window)
        X_daily.append(daily_window)
        y_entry.append(entry_labels[idx])
        y_direction.append(direction_labels[idx])

# 🔵 numpy array 변환
X_minute = np.array(X_minute, dtype=np.float32)
X_daily = np.array(X_daily, dtype=np.float32)
y_entry = np.array(y_entry, dtype=np.float32)
y_direction = np.array(y_direction, dtype=np.int64)

# 🔵 저장
np.save(os.path.join(save_dir, X_minute_file), X_minute)
np.save(os.path.join(save_dir, X_daily_file), X_daily)
np.save(os.path.join(save_dir, y_entry_file), y_entry)
np.save(os.path.join(save_dir, y_direction_file), y_direction)

print(f"✅ 저장 완료: X_minute: {X_minute.shape}, X_daily: {X_daily.shape}, y_entry: {y_entry.shape}")
