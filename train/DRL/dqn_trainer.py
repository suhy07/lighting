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
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
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
    def __init__(self, env, load_model=False):
        self.env = env
        self.state_dim = 21  # 根据_get_state()计算得到的状态维度
        self.action_dim = 9   # 动作空间大小
        
        # 检测GPU可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        
        if load_model:
            try:
                self.policy_net.load_state_dict(torch.load('models/thunder_dqn.pth', map_location=self.device))
                print("成功加载已有模型")
            except:
                print("未找到已有模型，将使用新模型开始训练")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.best_reward = float('-inf')
        
        self.buffer = ReplayBuffer(50000)  # 增加经验回放缓冲区大小
        self.batch_size = 256  # 进一步增大批处理大小
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # 提高最小探索率
        self.epsilon_decay = 0.997  # 降低探索率衰减速度

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return
        
        transitions = self.buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
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

    def save_model(self, path='models/', total_reward=None):
        if total_reward is not None and total_reward > self.best_reward:
            self.best_reward = total_reward
            torch.save(self.policy_net.state_dict(), path+'thunder_dqn.pth')
            print(f"保存新的最佳模型，奖励值: {total_reward:.2f}")

    def train(self, num_episodes=1000):
        episode_rewards = []
        best_reward = float('-inf')
        no_improvement_count = 0
        max_no_improvement = 20  # 如果20个episode没有提升就停止训练
        
        try:
            for episode in range(num_episodes):
                state = self.env.reset()
                total_reward = 0
                steps = 0
                
                while True:
                    action = self.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.buffer.push(state, action, reward, next_state, done)
                    
                    if len(self.buffer) >= self.batch_size:
                        self.update_model()
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    # 每100步更新一次目标网络
                    if steps % 100 == 0:
                        self.update_target_net()
                    
                    if done:
                        episode_rewards.append(total_reward)
                        # 更新训练状态显示
                        status_msg = f"\r训练进度：第{episode+1}/{num_episodes}轮 | 奖励：{total_reward:.2f} | epsilon：{self.epsilon:.3f}"
                        if total_reward > best_reward:
                            best_reward = total_reward
                            self.save_model(total_reward=total_reward)
                            no_improvement_count = 0
                            status_msg += f" | 新的最佳奖励！"
                        else:
                            no_improvement_count += 1
                        
                        print(status_msg, end="" if episode < num_episodes-1 else "\n")
                        
                        if no_improvement_count >= max_no_improvement:
                            print(f"\n训练在{episode+1}轮后停止，因为{max_no_improvement}轮内没有改进")
                            break
                        break
                
                if no_improvement_count >= max_no_improvement:
                    break
        
        except KeyboardInterrupt:
            print("\n训练被用户中断")
        
        finally:
            self.env.close()
            print(f"\n训练结束，最佳奖励: {best_reward:.2f}")


if __name__ == '__main__':
    env = ThunderGameEnv()
    trainer = DQNTrainer(env, load_model=True)
    trainer.train(num_episodes=1000)