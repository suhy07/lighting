import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        # 假设状态是一个经过特征提取的向量，尺寸为state_dim
        self.conv1 = nn.Conv2d(state_dim[0], hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        # 根据特征图的尺寸调整卷积层和池化层
        self.fc1 = nn.Linear(hidden_dim * (state_dim[1] // 2) * (state_dim[2] // 2), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出状态值

    def forward(self, features):
        x = torch.relu(self.conv1(features))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平卷积特征图
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x