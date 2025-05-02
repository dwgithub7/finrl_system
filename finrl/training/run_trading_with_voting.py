# run_trading_with_voting.py
import numpy as np

# 모듈 import
from finrl.training.trading_env import TradingEnv
from finrl.ensemble.dqn_agent import DQNAgent
from finrl.ensemble.ppo_agent import PPOAgent
from finrl.ensemble.voting_agent import VotingAgent

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 데이터 로드 (예시)
X_minute = np.load("data/dualbranch/SOLUSDT_1m_finrl_20250101~20250426_X_minute_full.npy")
X_daily = np.load("data/dualbranch/SOLUSDT_1m_finrl_20250101~20250426_X_daily_full.npy")
y_entry = np.load("data/dualbranch/SOLUSDT_1m_finrl_20250101~20250426_y_entry_full.npy")
y_direction = np.load("data/dualbranch/SOLUSDT_1m_finrl_20250101~20250426_y_direction_full.npy")

# 환경 초기화
env = TradingEnv(X_minute, X_daily, y_entry, y_direction)
state_dim = X_minute.shape[1] * X_minute.shape[2] + X_daily.shape[1] * X_daily.shape[2] + 1
action_dim = 4  # [0=관망, 1=롱, 2=숏, 3=청산]

# 개별 에이전트 초기화
dqn_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
ppo_agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
voting_agent = VotingAgent(dqn_agent, ppo_agent)

# 트레이딩 루프
num_episodes = 10
for ep in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = voting_agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    print(f"[EP {ep}] Total Reward: {total_reward:.2f}")
