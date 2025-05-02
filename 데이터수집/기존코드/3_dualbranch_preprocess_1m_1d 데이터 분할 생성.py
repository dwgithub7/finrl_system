import pandas as pd
import os

# 🔥 설정
input_file = "data/processed/SOLUSDT_1m_finrl_processed_20240101~20250426.csv"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# 🔵 입력 파일명에서 심볼+타임프레임+기간 분리
filename = os.path.basename(input_file).replace(".csv", "")
parts = filename.split("_")  # ['SOLUSDT', '1m', 'finrl', 'processed', '20240101~20250426']

symbol_tf = f"{parts[0]}_{parts[1]}"  # 'SOLUSDT_1m'
period = parts[-1]                   # '20240101~20250426'

# 🔵 출력 파일명 생성
minute_output = f"{symbol_tf}_with_tech_{period}.csv"
daily_output = f"{symbol_tf}_daily_with_tech_{period}.csv"

# 📥 로딩
df_minute = pd.read_csv(input_file)
df_minute['date'] = pd.to_datetime(df_minute['date'])
df_minute = df_minute.sort_values('date')

# ✅ 1분봉 저장
df_minute.to_csv(os.path.join(output_dir, minute_output), index=False)

# ✅ 1일봉 리샘플링
df_daily = df_minute.resample('1D', on='date').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'macd': 'last',
    'rsi': 'last'
}).dropna().reset_index()

# ✅ tic 추가
if 'tic' not in df_daily.columns:
    df_daily['tic'] = parts[0].replace('USDT', '')  # 예: SOLUSDT → SOL

df_daily.to_csv(os.path.join(output_dir, daily_output), index=False)

print(f"✅ 1m/1d 분할 저장 완료: {minute_output}, {daily_output}")
