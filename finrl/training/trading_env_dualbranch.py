# ✅ 실전용 TradingEnv 구축 + Random Agent 테스트

import numpy as np
import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, X_features, commission=0.0005, slippage=0.0005, initial_balance=10000):
        super(TradingEnv, self).__init__()

        self.X_features = X_features  # (num_steps, feature_dim)
        self.num_steps = len(X_features)
        self.feature_dim = X_features.shape[1]

        self.commission = commission
        self.slippage = slippage
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(4)  # 0:관망, 1:롱, 2:숏, 3:청산
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim + 1,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: 관망, 1: 롱, -1: 숏
        self.entry_price = None
        return self._get_observation()

    def step(self, action):
        reward = 0
        done = False

        current_feature = self.X_features[self.current_step]
        next_feature = self.X_features[self.current_step + 1] if self.current_step + 1 < self.num_steps else current_feature

        current_price = current_feature[0]  # 종가를 0번 컬럼이라고 가정
        next_price = next_feature[0]

        # ✅ 행동 처리
        if action == 1:  # 롱 진입
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
        elif action == 2:  # 숏 진입
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
        elif action == 3:  # 포지션 청산
            if self.position != 0 and self.entry_price is not None:
                pnl = self._calculate_pnl(current_price)
                reward += pnl
                self.balance += pnl
                self.position = 0
                self.entry_price = None

        # ✅ 슬리피지 + 수수료 반영
        reward -= (self.commission + self.slippage) * abs(self.position)

        # ✅ 최대 낙폭 체크
        if self.balance < self.initial_balance * 0.8:
            reward -= 1  # 추가 패널티

        # ✅ 관망 페널티
        if action == 0:
            reward -= 0.01

        # ✅ 스텝 이동
        self.current_step += 1
        if self.current_step >= self.num_steps - 1:
            done = True

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        feature = self.X_features[self.current_step]
        obs = np.append(feature, self.position)
        return obs

    def _calculate_pnl(self, exit_price):
        if self.position == 1:
            return (exit_price - self.entry_price) * (self.balance / self.entry_price)
        elif self.position == -1:
            return (self.entry_price - exit_price) * (self.balance / self.entry_price)
        else:
            return 0


# ✅ 테스트용 Random Agent
if __name__ == "__main__":
    # 예시 데이터: (1000스텝, 37개 피처)
    X_dummy = np.random.randn(1000, 37).astype(np.float32)
    X_dummy[:,0] += 100  # 종가 값 예시 (100 부근)

    env = TradingEnv(X_dummy)
    obs = env.reset()

    total_reward = 0
    done = False
    step_count = 0

    while not done:
        action = env.action_space.sample()  # 랜덤 행동 선택
        next_obs, reward, done, info = env.step(action)

        total_reward += reward
        step_count += 1

    print(f"✅ Random Agent 테스트 완료: 총 스텝 {step_count}, 총 Reward {total_reward:.2f}")
