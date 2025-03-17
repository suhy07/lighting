import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from environment import ThunderGameEnv

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNTrainer:
    def __init__(self, env):
        self.env = env
        self.state_dim = 21  # 根据_get_state()计算得到的状态维度
        self.action_dim = 9   # 动作空间大小
        
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
        self.buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return
        
        transitions = self.buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(batch[1])
        rewards = torch.FloatTensor(batch[2])
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.FloatTensor(batch[4])
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        expected_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path='models/'):
        torch.save(self.policy_net.state_dict(), path+'thunder_dqn.pth')

if __name__ == '__main__':
    env = ThunderGameEnv()
    trainer = DQNTrainer(env)
    
    episode_rewards = []
    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = trainer.select_action(state)
            next_state, reward, done, _ = env.step(action)
            trainer.buffer.push(state, action, reward, next_state, done)
            
            trainer.update_model()
            
            state = next_state
            total_reward += reward
            
            if done:
                episode_rewards.append(total_reward)
                print(f'Episode {episode}, Total reward: {total_reward:.2f}, Epsilon: {trainer.epsilon:.3f}')
                break
        
        if episode % 10 == 0:
            trainer.update_target_net()
            trainer.save_model()