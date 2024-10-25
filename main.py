import torch
from torch import nn

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim[0], hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(hidden_dim * input_dim[1], hidden_dim)
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
        self.conv1 = nn.Conv1d(input_dim[0], hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(hidden_dim * input_dim[1], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出状态值

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平卷积特征图
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 处理YOLOv5输出的函数
def process_yolov5_output(yolo_output, num_classes, threshold=0.5):
    batch_size, num_anchors, num_params = yolo_output.shape
    assert num_params == (4 + 1 + num_classes), "The number of parameters in the YOLOv5 output must match the expected number of parameters."

    bbox_xywh = yolo_output[:, :, :4]
    class_probs = torch.softmax(yolo_output[:, :, 5:], dim=2)
    class_scores, class_labels = torch.max(class_probs, dim=2)

    class_labels[class_scores < threshold] = -1  # 应用阈值

    features = torch.cat([bbox_xywh, class_labels.unsqueeze(2), class_scores.unsqueeze(2)], dim=2)
    features = features.view(batch_size, 6, -1)  # 6个特征

    return features

# 主程序
if __name__ == "__main__":
    yolo_output = torch.randn(1, 128520, 15)  # 示例输出
    num_classes = 10
    features = process_yolov5_output(yolo_output, num_classes)

    state_dim = (6, 128520)  # 特征图的尺寸
    # features = features.permute(0, 2, 1)  # 转换为 (batch_size, num_features, num_anchors)

    actor = Actor(input_dim=state_dim, action_dim=4)
    critic = Critic(input_dim=state_dim)

    action_probs = actor(features)
    state_value = critic(features)

    print("动作概率:", action_probs)
    print("状态值:", state_value)
