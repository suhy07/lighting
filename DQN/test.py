import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 环境配置
ENV_NAME = "YourEnvironment"
IMAGE_SIZE = (84, 84)  # 图片的尺寸
BATCH_SIZE = 32
GAMMA = 0.99  # 折扣因子
LEARNING_RATE = 0.001
EPISODES = 1000
MEMORY_CAPACITY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 200


# 检查是否存在文件夹，如果不存在则创建
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


# 图片预处理
def preprocess(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, IMAGE_SIZE)
    image = (image / 255.0).astype(np.float32)
    image = image[np.newaxis, ...]  # 添加批次维度
    return image


# 构建DQN模型
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 经验回放
class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


# 实例化DQN模型和目标网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(1, 2).to(device)  # 假设有2个动作
target_model = DQN(1, 2).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

# 实例化经验回放缓冲区
memory = ReplayBuffer(MEMORY_CAPACITY)

# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 训练过程
def train():
    epsilon = EPSILON_START
    for episode in range(EPISODES):
        state = preprocess("./Output/frame_0000.jpg").to(device)
        done = False
        while not done:
            # Epsilon-greedy exploration
            if random.random() < epsilon:
                action = torch.tensor([[random.randint(0, 1)]], dtype=torch.long).to(device)
            else:
                q_values = model(state)
                action = torch.argmax(q_values).unsqueeze(0).to(device)

            # 这里应该是与环境交互，获取下一个状态和奖励
            # next_state = ...
            # reward = ...
            next_state = preprocess("path_to_your_next_image_file").to(device)
            reward = torch.tensor([[random.randint(-1, 1)]], dtype=torch.float).to(device)
            done = random.choice([True, False])

            # 存储经验
            memory.add(state, action, reward, next_state, done)

            # 从记忆中采样并训练网络
            if memory.size() > BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                states = torch.cat(states)
                actions = torch.cat(actions)
                rewards = torch.cat(rewards)
                next_states = torch.cat(next_states)
                dones = torch.cat(dones)

                # 计算Q值和目标
                q_values = model(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    targets = rewards + GAMMA * next_q_values * (1 - dones)

                # 更新网络
                optimizer.zero_grad()
                loss = nn.MSELoss()(q_values, targets)
                loss.backward()
                optimizer.step()

            # 更新目标网络
            if episode % 10 == 0:
                target_model.load_state_dict(model.state_dict())

            state = next_state
            epsilon = max(epsilon * 0.99, EPSILON_END)


# 调用训练函数
train()
