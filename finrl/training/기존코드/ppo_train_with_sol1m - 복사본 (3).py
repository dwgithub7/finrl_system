import os
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gym

# CSV 파일 경로
DATA_PATH = "data/processed/sol1m_finrl_format.csv"
SAVE_MODEL_PATH = "data/models/ppo_sol_model"

# 1. 데이터 로드
df = pd.read_csv(DATA_PATH)

# 2. 필수 컬럼 추가 및 타입 변환
if 'tic' not in df.columns:
    df['tic'] = 'SOL'
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'])

# 3. Feature Engineering
tech_indicator_list = ['macd', 'rsi']
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=tech_indicator_list,
    use_turbulence=False,
    user_defined_feature=False
)
df = fe.preprocess_data(df)
print("✅ 전처리 완료. 컬럼:", df.columns.tolist())

# 4. 데이터 분할
df = df.sort_values("date")
train = data_split(df, start='2025-01-01', end='2025-04-23')
train = train.reset_index(drop=True)
print("✅ train 데이터 shape:", train.shape)

# 5. 환경 설정
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

# 6. 환경 인스턴스화
train_env = DummyVecEnv([lambda: Monitor(StockTradingEnv(df=train, **env_kwargs))])

# 7. DRL Agent 및 PPO 모델 생성
agent = DRLAgent(env=train_env)
ppo_params = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
    "device": "cuda"  # GPU 사용
}
model = agent.get_model("ppo", model_kwargs=ppo_params)

# 8. 모델 학습
trained_model = agent.train_model(model=model, tb_log_name="ppo", total_timesteps=10000)

# 9. 모델 저장
os.makedirs("models", exist_ok=True)
trained_model.save(SAVE_MODEL_PATH)
print("✅ PPO 모델 저장 완료:", SAVE_MODEL_PATH)
