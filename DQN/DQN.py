import torch
import torch.nn as nn
import torch.optim as optim


# DQN网络结构
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 超参数
state_size = 4  # 状态向量的维度
action_size = 2  # 假设有两个动作：左移和右移
epsilon = 0.1  # 探索率
alpha = 0.01  # 学习率
gamma = 0.99  # 折扣因子
batch_size = 32  # 批量大小

# 初始化网络和目标网络
dqn = DQN(state_size, action_size)
target_dqn = DQN(state_size, action_size)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()

# 优化器
optimizer = optim.Adam(dqn.parameters(), lr=alpha)


# 经验回放
class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        for i in batch:
            states.append(self.buffer[i][0])
            actions.append(self.buffer[i][1])
            rewards.append(self.buffer[i][2])
            next_states.append(self.buffer[i][3])
            dones.append(self.buffer[i][4])
        return states, actions, rewards, next_state
