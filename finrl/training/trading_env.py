import numpy as np
import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, X_minute, X_daily, y_entry, y_direction):
        super(TradingEnv, self).__init__()
        self.X_minute = X_minute
        self.X_daily = X_daily
        self.y_entry = y_entry
        self.y_direction = y_direction
        self.current_step = 0
        self.max_steps = len(y_entry)
        self.action_space = spaces.Discrete(4)  # 예: 0=관망, 1=매수, 2=매도, 3=청산
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X_minute.shape[1] + X_daily.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        reward = self._get_reward(action)

        # 먼저 done 여부 판단
        done = (self.current_step >= len(self.X_minute) - 1)

        # done이 아니면 다음 step
        if not done:
            self.current_step += 1
            obs = self._get_observation()
        else:
            obs = np.zeros_like(self.X_minute[0])  # dummy obs 리턴

        return obs, reward, done, {}


    def _get_observation(self):
        x_min = self.X_minute[self.current_step]
        x_day = self.X_daily[self.current_step]
        return np.concatenate([x_min, x_day])

    def _get_reward(self, action):
        if action == 1 and self.y_entry[self.current_step] == 1:
            return 1.0 if self.y_direction[self.current_step] == 1 else -1.0
        elif action == 2 and self.y_entry[self.current_step] == 1:
            return 1.0 if self.y_direction[self.current_step] == 0 else -1.0
        return -0.02  # 소극적 행동 패널티