import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 卷积层处理游戏状态
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 计算卷积层输出的特征维度
        self.conv_output_size = self._get_conv_output_size(state_size)
        
        # 全连接层
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_size)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.2)

    def _get_conv_output_size(self, shape):
        # 计算卷积层输出的特征维度
        x = torch.zeros(1, 1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(torch.prod(torch.tensor(x.size()[1:])))

    def forward(self, x):
        # 确保输入形状正确
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # 添加batch维度
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
            
        # 卷积层
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
