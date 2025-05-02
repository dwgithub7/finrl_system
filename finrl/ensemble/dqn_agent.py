import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = QNetwork(state_size, action_size).to(self.device)

        self.target_model = QNetwork(state_size, action_size).to(self.device)  # ✅ 추가
        self.target_model.load_state_dict(self.model.state_dict())  # ✅ 초기 가중치 동기화
        self.target_model.eval()  # ✅ 학습 안함 설정

        self.update_target_freq = 10  # ✅ 주기 설정 (에피소드 기준)
        self.train_step = 0  # ✅ 학습 step 카운터

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).reshape(-1).unsqueeze(0).to(self.device)  # ✅ flatten → (1, 800)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # 🔁 [1] 상태 및 보상 데이터 처리 (flatten 적용)
        states = torch.FloatTensor([m[0].reshape(-1) for m in minibatch]).to(self.device)          # (batch, 800)
        actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1).to(self.device)         # (batch, 1)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).unsqueeze(1).to(self.device)        # (batch, 1)
        next_states = torch.FloatTensor([m[3].reshape(-1) for m in minibatch]).to(self.device)     # (batch, 800)
        dones = torch.FloatTensor([float(m[4]) for m in minibatch]).unsqueeze(1).to(self.device)   # (batch, 1)

        # 🔁 [2] 현재 Q값 계산
        q_values = self.model(states).gather(1, actions)  # (batch, 1)

        # 🔁 [3] 다음 상태에서의 최대 Q값 (Target Network)
        next_q_values = self.target_model(next_states).max(1)[0].detach().unsqueeze(1)  # (batch, 1)

        # 🔁 [4] 타겟 Q값 계산
        targets = rewards + (1 - dones) * self.gamma * next_q_values  # (batch, 1)

        # 🔁 [5] 손실 계산 및 역전파
        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()  # 🔄 Learning Rate Scheduler 적용

        # 🔁 [6] Soft Update 적용
        self.soft_update_target_network(tau=0.01)




    def soft_update_target_network(self, tau=0.01):
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    def save(self, path):
        torch.save(self.model.state_dict(), path)