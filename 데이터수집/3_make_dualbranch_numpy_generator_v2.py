# make_dualbranch_numpy_generator_v2.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ✅ 입력 파일
INPUT_FILE = "data/processed/SOLUSDT_1m_finrl_processed_20240101~20250426.csv"

# ✅ 출력 디렉토리
OUTPUT_DIR = "data/dualbranch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ 파라미터 설정
WINDOW_MINUTE = 30
WINDOW_DAILY = 10
HORIZON = 1
ENTRY_THRESHOLD = 0.001

# ✅ 파일명 파싱
basename = os.path.basename(INPUT_FILE).replace(".csv", "")  # SOLUSDT_1m_finrl_processed_20240101~20250426
symbol_tf_period = "_".join(basename.split("_")[:3] + [basename.split("_")[-1]])  # SOLUSDT_1m_20240101~20250426

# ✅ 데이터 로드
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])

# ✅ 1일봉 데이터 생성
df_daily = df.resample('1D', on='date').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'macd': 'last',
    'rsi': 'last',
    'cci': 'last',
    'adx': 'last',
    'atr': 'last',
    'obv': 'last',
    'ma5': 'last',
    'ma10': 'last',
    'ma20': 'last',
    'body_size': 'last',
    'shadow_ratio': 'last'
}).dropna().reset_index()

print(f"✅ 1일봉 데이터 생성 완료: {df_daily.shape}")

# ✅ 1분봉과 1일봉 매핑 준비
df['date_only'] = df['date'].dt.date
df_daily['date_only'] = df_daily['date'].dt.date
daily_dict = df_daily.set_index('date_only').to_dict(orient='index')

# ✅ 샘플 생성
X_minute = []
X_daily = []
y_entry = []
y_direction = []

for idx in tqdm(range(WINDOW_MINUTE, len(df) - HORIZON)):
    minute_seq = df.iloc[idx-WINDOW_MINUTE:idx]
    sample_date = minute_seq.iloc[-1]['date_only']
    
    daily_seq = []
    for i in range(WINDOW_DAILY):
        prev_date = sample_date - pd.Timedelta(days=WINDOW_DAILY - i - 1)
        if prev_date in daily_dict:
            daily_values = [v for k, v in daily_dict[prev_date].items() if k != 'date']  # ✅ 날짜 제외
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

    X_minute.append(minute_seq[['close', 'volume', 'macd', 'rsi', 'cci', 'adx', 'atr', 'obv', 'body_size', 'shadow_ratio']].values)
    X_daily.append(np.array(daily_seq))
    y_entry.append(entry)
    y_direction.append(direction)


print(f"✅ 데이터 생성 완료: {len(X_minute)} samples")

# ✅ Numpy 저장
np.save(os.path.join(OUTPUT_DIR, f"{symbol_tf_period}_X_minute.npy"), np.array(X_minute, dtype=np.float32))
np.save(os.path.join(OUTPUT_DIR, f"{symbol_tf_period}_X_daily.npy"), np.array(X_daily, dtype=np.float32))
np.save(os.path.join(OUTPUT_DIR, f"{symbol_tf_period}_y_entry.npy"), np.array(y_entry, dtype=np.float32))
np.save(os.path.join(OUTPUT_DIR, f"{symbol_tf_period}_y_direction.npy"), np.array(y_direction, dtype=np.int64))

print(f"✅ Numpy 저장 완료: {OUTPUT_DIR}")
