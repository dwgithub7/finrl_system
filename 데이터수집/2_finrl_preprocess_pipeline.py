import pandas as pd
import os
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

# 경로 설정
input_file = "data/raw/SOLUSDT_1m_raw_20250101~20250426.csv"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

filename = os.path.basename(input_file).replace(".csv", "")
parts = filename.split("_")
symbol_tf = f"{parts[0]}_{parts[1]}"
period = parts[-1]
output_file = f"{symbol_tf}_finrl_processed_{period}.csv"

# 데이터 로드
df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'])
df['tic'] = parts[0].replace('USDT', '')

# ✅ obv 직접 생성
df['obv'] = (df['volume'] * ((df['close'].diff() > 0).astype(int) - (df['close'].diff() < 0).astype(int))).cumsum()
df['obv'] = df['obv'].fillna(0)

# ✅ body_size, shadow_ratio 직접 생성
df['body_size'] = (df['close'] - df['open']) / df['open']
df['shadow_ratio'] = (df['high'] - df['low']) / df['close']

# ✅ ma5, ma10, ma20 직접 생성
df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()

# ✅ FeatureEngineer로 나머지 기술 지표 추가
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=["macd", "rsi", "cci", "adx", "atr"],  # ma5, ma10, ma20은 제외
    use_turbulence=False,
    user_defined_feature=False
)
df = fe.preprocess_data(df)

# 저장
save_path = os.path.join(output_dir, output_file)
df.to_csv(save_path, index=False)

print(f"✅ 전처리 및 저장 완료: {save_path} (shape={df.shape})")

