import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Actor网络，用于选择动作
class Actor(nn.Module):
    def __init__(self, state_size, action_size=8, hidden_size=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, action_size),
            nn.Tanh()  # 使用Tanh激活函数输出连续动作
        )
    
    def forward(self, state):
        return self.net(state)

# Critic网络，用于评估状态价值
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        return self.net(state)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(state).to(device),
            torch.LongTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

# DRL训练器
class ThunderDRLTrainer:
    def __init__(self, env, state_size, action_size,
                 actor_lr=1e-5, critic_lr=1e-4,
                 gamma=0.99, buffer_size=100000,
                 batch_size=64, update_every=4):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = 256  # 增大批处理大小以提高GPU利用率
        self.update_every = update_every
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')
        
        # 初始化网络
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)
        
        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 初始化经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)
        
        # 训练步数计数器
        self.t_step = 0
        
        # 早停相关参数
        self.best_reward = 0.0  # 设置一个基准值
        self.patience = 5000  # 增加耐心值以给予更多学习机会
        self.patience_counter = 0
        
        # epsilon衰减参数
        self.epsilon = 1.0
        self.epsilon_min = 0.15  # 提高最小探索率以保持探索性
        self.epsilon_decay = 0.9998  # 降低衰减速率以延长探索时间
    
    def step(self, state, action, reward, next_state, done):
        # 保存经验到回放缓冲区
        self.memory.add(state, action, reward, next_state, done)
        
        # 每update_every步学习一次
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.learn()
    
    def act(self, state, use_epsilon=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action_probs = self.actor(state)
        self.actor.train()
        
        # epsilon-greedy策略
        if use_epsilon and random.random() < self.epsilon:
            action = random.choice(np.arange(self.action_size))
        else:
            action = np.argmax(action_probs.cpu().numpy())
        
        # 更新epsilon
        if use_epsilon:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return action
    
    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算移动和射击的奖励
        base_reward = 5.0  # 提高基础奖励
        movement_reward = torch.abs(actions.float()).mean() * 2.0  # 增加移动奖励权重
        shooting_reward = (actions != 0).float().mean() * 2.5  # 增加射击奖励权重
        combined_rewards = rewards + base_reward + movement_reward + shooting_reward
        
        # 更新Critic
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            target_values = combined_rewards + (1 - dones) * self.gamma * next_values
        
        current_values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(current_values, target_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        actions_pred = self.actor(states)
        advantages = (target_values - current_values).detach()
        
        # 使用MSE损失计算连续动作的策略梯度
        actions_tensor = actions.unsqueeze(1).float()
        actor_loss = torch.mean(torch.pow(actions_pred - actions_tensor, 2) * advantages.unsqueeze(1))
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 早停检查
        mean_reward = combined_rewards.mean().item()
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.patience_counter = 0
            self.save()  # 保存最佳模型
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered. Best reward: {self.best_reward}")
                return True  # 返回True表示应该停止训练
        return False  # 继续训练
    
    def save(self, path='./models'):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), os.path.join(path, 'thunder_actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'thunder_critic.pth'))
    
    def load(self, path='./models'):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'thunder_actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'thunder_critic.pth')))