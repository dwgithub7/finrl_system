import pandas as pd
import numpy as np
import os

# 🔥 설정
input_minute_file = "data/processed/SOLUSDT_1m_with_tech_20240101~20250426.csv"
input_daily_file = "data/processed/SOLUSDT_1m_daily_with_tech_20240101~20250426.csv"
save_dir = "data/numpy"
os.makedirs(save_dir, exist_ok=True)

# 🔵 파일명에서 심볼+타임프레임+기간 추출
def extract_info(file_path):
    filename = os.path.basename(file_path).replace(".csv", "")
    parts = filename.split("_")  # ['SOLUSDT', '1m', 'with', 'tech', '20240101~20250426']
    symbol_tf = f"{parts[0]}_{parts[1]}"
    period = parts[-1]
    return symbol_tf, period

symbol_tf, period = extract_info(input_minute_file)

# 🔵 출력 파일명 생성
X_minute_file = f"{symbol_tf}_X_minute_{period}.npy"
X_daily_file = f"{symbol_tf}_X_daily_{period}.npy"
y_entry_file = f"{symbol_tf}_y_entry_{period}.npy"
y_direction_file = f"{symbol_tf}_y_direction_{period}.npy"

# 📥 데이터 로딩
df_minute = pd.read_csv(input_minute_file)
df_minute['date'] = pd.to_datetime(df_minute['date'])
df_minute = df_minute.sort_values('date')

df_daily = pd.read_csv(input_daily_file)
df_daily['date'] = pd.to_datetime(df_daily['date'])
df_daily = df_daily.sort_values('date')

# ✅ future return 계산
df_minute['future_return'] = df_minute['close'].shift(-1) / df_minute['close'] - 1

# ✅ 라벨링
threshold = 0.0002  # 예: 0.02% 수익 목표
df_minute['entry_label'] = (df_minute['future_return'].abs() > threshold).astype(int)
df_minute['direction_label'] = (df_minute['future_return'] > 0).astype(int)

# ✅ X 생성
feature_cols_minute = ["open", "high", "low", "close", "volume", "macd", "rsi"]
feature_cols_daily = ["open", "high", "low", "close", "volume", "macd", "rsi"]

WINDOW_SIZE_MIN = 30  # 30분
WINDOW_SIZE_DAY = 10   # 10일

X_minute = []
X_daily = []
y_entry = []
y_direction = []

for idx in range(max(WINDOW_SIZE_MIN, WINDOW_SIZE_DAY), len(df_minute)-1):
    end_time = df_minute.iloc[idx]['date']
    
    minute_window = df_minute.iloc[idx-WINDOW_SIZE_MIN:idx][feature_cols_minute].values
    daily_window = df_daily[df_daily['date'] < end_time].iloc[-WINDOW_SIZE_DAY:][feature_cols_daily].values

    if minute_window.shape[0] == WINDOW_SIZE_MIN and daily_window.shape[0] == WINDOW_SIZE_DAY:
        X_minute.append(minute_window)
        X_daily.append(daily_window)
        y_entry.append(df_minute.iloc[idx]['entry_label'])
        y_direction.append(df_minute.iloc[idx]['direction_label'])

# ✅ numpy 변환
X_minute = np.array(X_minute, dtype=np.float32)
X_daily = np.array(X_daily, dtype=np.float32)
y_entry = np.array(y_entry, dtype=np.float32)
y_direction = np.array(y_direction, dtype=np.int64)

# ✅ 저장
np.save(os.path.join(save_dir, X_minute_file), X_minute)
np.save(os.path.join(save_dir, X_daily_file), X_daily)
np.save(os.path.join(save_dir, y_entry_file), y_entry)
np.save(os.path.join(save_dir, y_direction_file), y_direction)

print(f"✅ 저장 완료: {X_minute.shape}, {X_daily.shape}, {y_entry.shape}, {y_direction.shape}")

