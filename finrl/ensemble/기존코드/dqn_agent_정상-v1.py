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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()


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
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).view(1, -1).to(self.device)
            next_state = torch.FloatTensor(next_state).view(1, -1).to(self.device)
            reward = torch.tensor(reward).to(self.device)
            done = torch.tensor(done).to(self.device)

            # print("🧠 model.device:", next(self.model.parameters()).device)
            # print("📦 state.device:", state.device)
            # print("📦 next_state.device:", next_state.device)

            q_next = torch.max(self.model(next_state))
            target = reward + (1 - done.float()) * self.gamma * q_next

            target_f = self.model(state)
            target_f = target_f.clone()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        # Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def save(self, path):
        torch.save(self.model.state_dict(), path)