import numpy as np
from finrl.training.trading_env import TradingEnv
from finrl.ensemble.dqn_agent import DQNAgent

import torch
import os
from torch.utils.data import DataLoader, TensorDataset

# 실전 데이터 경로
DATA_DIR = "data/dualbranch"
X_minute = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_X_minute_full.npy"))
X_daily = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_X_daily_full.npy"))
y_entry = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_y_entry_full.npy"))
y_direction = np.load(os.path.join(DATA_DIR, "SOLUSDT_1m_finrl_20250101~20250426_y_direction_full.npy"))

# 환경 초기화
env = TradingEnv(X_minute, X_daily, y_entry, y_direction)
# state_size = env.observation_space.shape[0]
state_size = X_minute.shape[1] * X_minute.shape[2] + X_daily.shape[1] * X_daily.shape[2]
# 즉: 30*20 + 10*20 = 800action_size = env.action_space.n
action_size = 4

# 에이전트 초기화
agent = DQNAgent(state_size=state_size, action_size=action_size)

print("✅ X_minute shape:", X_minute.shape)
print("✅ X_daily shape:", X_daily.shape)
print("✅ state_size:", state_size)
print("✅ model input dim:", agent.model.model[0].in_features)

# 학습 파라미터
episodes = 300  # 조절필요

agent.memory.clear()  # 🧹 메모리 초기화 추가 (기존에 잘못 저장된 데이터 제거)

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    agent.replay()
    print(f"EP {ep + 1}/{episodes}, Total Reward: {total_reward:.2f}")

# 모델 저장
agent.save("models/dqn_model.pt")