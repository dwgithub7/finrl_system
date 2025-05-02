import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# ✅ 경로 설정
DATA_PATH = "data/processed/SOLUSDT_1m_finrl_processed_20250101~20250426.csv"
OUTPUT_DIR = "data/dualbranch/"

WINDOW_MINUTE = 30
WINDOW_DAILY = 10
HORIZON = 1
ENTRY_THRESHOLD = 0.001

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df.columns = df.columns.str.strip()

# ✅ 파생 feature 추가
df['volume_ratio'] = df['volume'] / (df['volume'].rolling(window=5).mean() + 1e-6)
df['hlc_mean'] = (df['high'] + df['low'] + df['close']) / 3
df['volatility'] = (df['high'] - df['low']) / (df['close'] + 1e-6)
df['ma60'] = df['close'].rolling(window=60).mean()
df['ma120'] = df['close'].rolling(window=120).mean()

df.fillna(0, inplace=True)

# ✅ 1분봉 feature 목록
minute_features = [
    'open', 'high', 'low', 'close', 'volume',
    'obv', 'body_size', 'shadow_ratio',
    'ma5', 'ma10', 'ma20', 'macd', 'rsi', 'cci', 'adx', 'atr',
    'volume_ratio'
]

# ✅ 1분봉 스케일링
scaler = MinMaxScaler()
df[minute_features] = scaler.fit_transform(df[minute_features])

# ✅ 1일봉 데이터
df_daily = df.resample('1D', on='date').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
    'macd': 'last', 'rsi': 'last', 'cci': 'last', 'adx': 'last', 'atr': 'last', 'obv': 'last',
    'ma5': 'last', 'ma10': 'last', 'ma20': 'last', 'ma60': 'last', 'ma120': 'last',
    'body_size': 'last', 'shadow_ratio': 'last', 'volume_ratio': 'last', 'volatility': 'last'
}).dropna().reset_index()

df_daily.fillna(0, inplace=True)

# ✅ 1일봉 스케일링
daily_features = df_daily.columns.drop(['date', 'date_only'], errors='ignore')
df_daily[daily_features] = scaler.fit_transform(df_daily[daily_features])

df['date_only'] = df['date'].dt.date
df_daily['date_only'] = df_daily['date'].dt.date
df_daily = df_daily.drop(columns=['date'])
daily_dict = df_daily.set_index('date_only').to_dict(orient='index')

# ✅ 데이터 생성
X_minute, X_daily, y_entry, y_direction = [], [], [], []

for idx in tqdm(range(WINDOW_MINUTE, len(df) - HORIZON)):
    minute_seq = df.iloc[idx-WINDOW_MINUTE:idx]
    sample_date = minute_seq.iloc[-1]['date_only']

    daily_seq = []
    for i in range(WINDOW_DAILY):
        prev_date = sample_date - pd.Timedelta(days=WINDOW_DAILY - i - 1)
        if prev_date in daily_dict:
            daily_values = list(daily_dict[prev_date].values())
            daily_seq.append(daily_values)
        else:
            break

    if len(daily_seq) != WINDOW_DAILY:
        continue

    future_close = df.iloc[idx+HORIZON-1]['close']
    current_close = df.iloc[idx-1]['close']
    future_return = (future_close / current_close) - 1

    entry = 1 if abs(future_return) >= ENTRY_THRESHOLD else 0
    direction = 1 if future_return >= 0 else 0

    X_minute.append(minute_seq[minute_features].astype(np.float32).values)
    X_daily.append(np.array(daily_seq, dtype=np.float32))
    y_entry.append(entry)
    y_direction.append(direction)

X_minute = np.array(X_minute, dtype=np.float32)
X_daily = np.array(X_daily, dtype=np.float32)
y_entry = np.array(y_entry, dtype=np.float32)
y_direction = np.array(y_direction, dtype=np.int64)

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "SOLUSDT_1m_finrl_20250101~20250426_X_minute_full.npy"), X_minute)
np.save(os.path.join(OUTPUT_DIR, "SOLUSDT_1m_finrl_20250101~20250426_X_daily_full.npy"), X_daily)
np.save(os.path.join(OUTPUT_DIR, "SOLUSDT_1m_finrl_20250101~20250426_y_entry_full.npy"), y_entry)
np.save(os.path.join(OUTPUT_DIR, "SOLUSDT_1m_finrl_20250101~20250426_y_direction_full.npy"), y_direction)

print("✅ (최종 안정화) 데이터 생성 완료!")
