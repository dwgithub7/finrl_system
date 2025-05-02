
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
import gym




# ✅ 1. 사용자 CSV 데이터 불러오기
df = pd.read_csv("data/processed/sol1m_finrl_processed.csv")

# 'tic' 열 추가
df['tic'] = 'SOL'

# 'date' 열을 datetime 형식으로 변환
df['date'] = pd.to_datetime(df['date'])

# 기술 지표 목록 정의
tech_indicator_list = ['macd', 'rsi', 'cci', 'adx']


# ✅ 2. Feature Engineer 적용 (필요 시 지표 선택 수정 가능)
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    use_vix=False,
    use_turbulence=False,
    user_defined_feature=False,
)
df = fe.preprocess_data(df)

# ✅ 3. 훈련/테스트 데이터 분할
train = df[df.date < "2025-04-01"]
trade = df[df.date >= "2025-04-01"]

# ✅ 4. 환경 파라미터 정의
stock_dimension = len(train.tic.unique())

# 종목 수
stock_dim = len(df['tic'].unique())

# 상태 공간 계산
state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim


# 환경 설정
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

# ✅ 5. 학습용 환경 초기화
train_env = StockTradingEnv(df=train, **env_kwargs)
train_gym = gym.make('StockTradingEnv-v0', env=train_env)

# ✅ 6. PPO 에이전트 구성 및 훈련
agent = DRLAgent(env=train_gym)
ppo_model = agent.get_model("ppo")

trained_ppo = agent.train_model(model=ppo_model, tb_log_name="ppo", total_timesteps=50000)

# ✅ 7. 저장
trained_ppo.save("ppo_trained_sol1m")
print("✅ PPO 훈련 완료 및 저장됨: ppo_trained_sol1m.zip")
