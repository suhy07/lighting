import numpy as np
import torch
import torch.optim as optim

from DQN import DQN
from ReplayBuffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon, buffer_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建主网络和目标网络
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 经验回放缓冲区
        self.buffer = ReplayBuffer(buffer_size)
        
        # 目标网络更新频率
        self.target_update_freq = 10
        self.update_counter = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()

    def update(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # 转换为tensor并移动到设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失并更新
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
