import torch
from torch import nn, optim
import numpy as np
import random
from collections import deque
from tqdm import tqdm  # 引入进度条库


# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim[0], 128, kernel_size=3, stride=1, padding=1)  # 输出通道数为128
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * input_dim[1], 128)  # 输入维度根据卷积层输出计算
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), 128 * x.size(2))  # 保留通道数，并展平长度
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv1d(input_dim[0], 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * input_dim[1], 128)  # 输入维度根据卷积层输出计算
        self.fc2 = nn.Linear(128, 1)  # 输出状态值

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), 128 * x.size(2))  # 保留通道数，并展平长度
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# 经验回放类
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# 训练函数
def train(actor, critic, optimizer_actor, optimizer_critic, replay_buffer, num_episodes=1000, gamma=0.99,
          batch_size=64):
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state = torch.randn(1, 6, 128520)  # 初始化状态
        done = False
        total_reward = 0

        while not done:
            action_probs = actor(state)
            action = np.random.choice(len(action_probs[0]), p=action_probs.detach().numpy()[0])

            # 模拟奖励和下一个状态
            reward = random.uniform(-1, 1)
            next_state = torch.randn(1, 6, 128520)  # 示例下一个状态
            done = random.random() < 0.1  # 随机结束条件

            total_reward += reward

            # 保存经验
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state

            # 更新网络
            if replay_buffer.size() > batch_size:
                # 从经验回放中采样
                experiences = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*experiences)

                states = torch.cat(states)
                next_states = torch.cat(next_states)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                # 计算Critic的目标
                target = rewards + (1 - dones) * gamma * critic(next_states)
                values = critic(states)

                # 更新Critic
                critic_loss = nn.MSELoss()(values, target.detach())
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                # 更新Actor
                advantage = target - values.detach()
                actor_loss = -torch.log(action_probs[0][action]) * advantage
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")


# 保存模型参数
def save_model(actor, critic, actor_path='actor.pth', critic_path='critic.pth'):
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    print("模型参数已保存。")


# 加载模型参数
def load_model(actor, critic, actor_path='actor.pth', critic_path='critic.pth'):
    actor.load_state_dict(torch.load(actor_path))
    critic.load_state_dict(torch.load(critic_path))
    print("模型参数已加载。")


# 主程序
if __name__ == "__main__":
    state_dim = (6, 128520)
    action_dim = 4

    actor = Actor(input_dim=state_dim, action_dim=action_dim)
    critic = Critic(input_dim=state_dim)

    optimizer_actor = optim.Adam(actor.parameters(), lr=0.001)
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)

    replay_buffer = ReplayBuffer(max_size=10000)

    train(actor, critic, optimizer_actor, optimizer_critic, replay_buffer, num_episodes=1000)

    # 保存模型参数
    save_model(actor, critic)

    # 加载模型参数（如果需要）
    # load_model(actor, critic)
