import torch
from torch import optim, nn
import numpy as np
from lab.lighting.DRL.yolov5 import extract_features, model


def process_yolov5_output(yolo_output, threshold=0.5):
    batch_size, num_anchors, num_params = yolo_output.shape

    # 提取边界框的位置信息 (x, y, w, h)
    bbox_xywh = yolo_output[:, :, :4]

    # 提取类别概率，并使用softmax获取最可能的类别和置信度
    class_probs = torch.softmax(yolo_output[:, :, 5:], dim=2)
    class_scores, class_labels = torch.max(class_probs, dim=2)

    # 将类别概率转换为类别索引，未识别的类别用-1表示
    class_labels[class_labels < threshold] = -1

    # 将位置信息、置信度和最可能的类别概率合并为一个特征图
    features = torch.cat([bbox_xywh, class_labels.unsqueeze(2), class_scores.unsqueeze(2)], dim=2)

    # 将特征图的形状调整为 (batch_size, channels, height, width)
    num_features = 6  # xywh + class label + class score
    features = features.view(batch_size, num_anchors, num_features)

    return features

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(input_dim[0], hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(hidden_dim * input_dim[1] * input_dim[2], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平卷积特征图
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)  # 使用softmax获取动作概率
        return x

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(input_dim[0], hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(hidden_dim * input_dim[1] * input_dim[2], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出状态值

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平卷积特征图
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

image_path = r'..\image\frame_0007.jpg'  # 替换为你的图片路径
features = extract_features(image_path, model, img_size=(1920, 1920))

yolo_output = features  # YOLOv5模型的输出
num_classes = 10  # 类别数量

# 处理YOLOv5输出以获取特征向量
features = process_yolov5_output(yolo_output)

# 假设特征图的尺寸为 (1920, 1080)，这里我们使用了一个假设的尺寸
state_dim = (6, 1920, 1080)  # 使用YOLOv5输出的特征数作为特征图的通道数

# 创建Actor和Critic网络实例
actor = Actor(state_dim, action_dim=4)
critic = Critic(state_dim)

# 将特征向量作为输入传递给网络
action_probs = actor(features)
state_value = critic(features)

# env = gym.make('YourEnvironmentName')  # 替换为你的环境名称
state_dim = (3, 1080, 1920)  # 替换为你的状态维度
action_dim = 4  # 替换为你的动作维度

# 实例化Actor和Critic网络
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
# 定义优化器
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# 定义损失函数
criterion = nn.MSELoss()


# 训练循环
def train(actor, critic, actor_optimizer, critic_optimizer, episodes, batch_size):
    for episode in range(episodes):
        # 初始化状态
        state = env.reset()
        done = False
        episode_reward = 0
        episode_states, episode_actions, episode_rewards = [], [], []

        while not done:
            # Actor选择动作
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = actor(state_tensor)
            action = np.argmax(action_probs.cpu().numpy())  # 选择概率最高的动作

            # 与环境交互
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # 存储经验
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            # 更新状态
            state = next_state

        # 转换经验为张量
        states = torch.tensor(episode_states, dtype=torch.float32)
        actions = torch.tensor(episode_actions, dtype=torch.int64)
        rewards = torch.tensor(episode_rewards, dtype=torch.float32)

        # 更新Critic网络
        with torch.no_grad():
            next_values = critic(torch.tensor([next_state], dtype=torch.float32))
            target_values = rewards + (1 - done) * 0.99 * next_values

        current_values = critic(states)
        critic_loss = criterion(current_values, target_values)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # 更新Actor网络
        actor_loss = -critic(states).detach() * torch.log(action_probs)
        actor_loss = actor_loss.sum()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 打印信息
        if episode % 100 == 0:
            print(f'Episode {episode}, Reward: {episode_reward}')


# 初始化环境和网络
env = gym.make('YourEnvironmentName')  # 替换为你的环境名称
state_dim = env.observation_space.shape  # 替换为你的状态维度
action_dim = env.action_space.n  # 替换为你的动作维度
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)

# 开始训练
train(actor, critic, actor_optimizer, critic_optimizer, episodes=1000, batch_size=64)
