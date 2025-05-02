import os
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO

# CSV 파일 경로
DATA_PATH = "data/processed/sol1m_finrl_format.csv"
SAVE_MODEL_PATH = "data/models/ppo_sol_model"

df = pd.read_csv(DATA_PATH)

# 필수 컬럼 추가
if 'tic' not in df.columns:
    df['tic'] = 'SOL'
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'])

# 기술 지표 정의 (간단히 시작)
tech_indicator_list = ['macd', 'rsi']

# 전처리
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=tech_indicator_list,
    use_turbulence=False,
    user_defined_feature=False
)

df = fe.preprocess_data(df)
print("✅ 전처리 완료. 컬럼:", df.columns.tolist())

# 데이터 분할
train = data_split(df, start='2025-01-01', end='2025-04-23')
train = train.reset_index(drop=True)

# 환경 구성
stock_dim = len(train['tic'].unique())
state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": [0.001] * stock_dim,
    "sell_cost_pct": [0.001] * stock_dim,
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": tech_indicator_list,
    "action_space": stock_dim,
    "reward_scaling": 1e-4,
    "num_stock_shares": [0] * stock_dim
}

train_env = StockTradingEnv(df=train, **env_kwargs)

# DRL 에이전트 구성 및 PPO 학습
agent = DRLAgent(env=train_env)
model = agent.get_model("ppo", policy="MlpPolicy")

trained_model = agent.train_model(model=model, tb_log_name="ppo", total_timesteps=10000)

# 모델 저장
os.makedirs("models", exist_ok=True)
trained_model.save(SAVE_MODEL_PATH)
print("✅ PPO 모델 저장 완료:", SAVE_MODEL_PATH)