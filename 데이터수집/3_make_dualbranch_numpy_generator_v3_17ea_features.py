import pandas as pd
import numpy as np
import os

# ✅ 경로 설정
INPUT_FILE = "data/processed/SOLUSDT_1m_finrl_processed_20240101~20250426.csv"
OUTPUT_DIR = "data/dualbranch"

# ✅ 하이퍼파라미터 설정
minute_seq_len = 30
daily_seq_len = 10

# ✅ 데이터 로드
df = pd.read_csv(INPUT_FILE)

# ✅ 17개 전체 Feature 사용 (순서 유지)
feature_columns = [
    'open', 'high', 'low', 'close', 'volume', 'tic', 'obv', 'body_size', 'shadow_ratio',
    'ma5', 'ma10', 'ma20', 'macd', 'rsi', 'cci', 'adx', 'atr'
]

# ✅ X_minute, X_daily, y_entry, y_direction 리스트 생성
X_minute = []
X_daily = []
y_entry = []
y_direction = []

# ✅ 시간순으로 데이터 정렬 (필수)
df = df.sort_values(by=['tic', 'date'])

# ✅ 종목별로 그룹화
groups = df.groupby('tic')

for tic, group in groups:
    group = group.reset_index(drop=True)

    for i in range(minute_seq_len, len(group) - daily_seq_len):
        minute_seq = group.iloc[i-minute_seq_len:i]
        daily_seq = group.iloc[i:i+daily_seq_len]

        # ✅ 17개 Feature 모두 사용
        X_minute.append(minute_seq[feature_columns].values)
        X_daily.append(daily_seq[feature_columns].values)

        # ✅ 레이블 정의 (예시: 다음 step close 가격 비교)
        entry_label = 1 if group.iloc[i+1]['close'] > group.iloc[i]['close'] else 0
        direction_label = 1 if group.iloc[i+1]['close'] > group.iloc[i]['close'] else 0

        y_entry.append(entry_label)
        y_direction.append(direction_label)

# ✅ numpy 배열 변환
X_minute = np.array(X_minute)
X_daily = np.array(X_daily)
y_entry = np.array(y_entry)
y_direction = np.array(y_direction)

# ✅ 저장 (파일명에 _full 추가)
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "SOLUSDT_1m_finrl_20240101~20250426_X_minute_full.npy"), X_minute)
np.save(os.path.join(OUTPUT_DIR, "SOLUSDT_1m_finrl_20240101~20250426_X_daily_full.npy"), X_daily)
np.save(os.path.join(OUTPUT_DIR, "SOLUSDT_1m_finrl_20240101~20250426_y_entry.npy"), y_entry)
np.save(os.path.join(OUTPUT_DIR, "SOLUSDT_1m_finrl_20240101~20250426_y_direction.npy"), y_direction)

print("✅ Numpy 파일 생성 완료!")
