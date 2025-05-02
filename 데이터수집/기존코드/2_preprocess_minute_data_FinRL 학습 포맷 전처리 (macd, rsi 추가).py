import pandas as pd
import os
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

# 🔥 설정
input_file = "data/raw/SOLUSDT_1m_raw_20240101~20250426.csv"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# 🔵 입력 파일명에서 심볼+타임프레임+기간 분리
filename = os.path.basename(input_file).replace(".csv", "")
parts = filename.split("_")  # ['SOLUSDT', '1m', 'raw', '20240101~20250426']

symbol_tf = f"{parts[0]}_{parts[1]}"  # 'SOLUSDT_1m'
period = parts[-1]                   # '20240101~20250426'

# 🔵 출력 파일명 생성
output_file = f"{symbol_tf}_finrl_processed_{period}.csv"

# 📥 로딩
df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'])

# ✅ tic 추가
if 'tic' not in df.columns:
    df['tic'] = parts[0].replace('USDT', '')  # 예: SOLUSDT → SOL

# ✅ 기술지표 추가
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=["macd", "rsi"],
    use_turbulence=False,
    user_defined_feature=False
)
df = fe.preprocess_data(df)

# 📤 저장
save_path = os.path.join(output_dir, output_file)
df.to_csv(save_path, index=False)

print(f"✅ 전처리 완료 및 저장: {save_path} (shape: {df.shape})")
