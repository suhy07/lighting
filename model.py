import torch
from torch import nn, optim
import numpy as np
import random
from collections import deque
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define Actor network
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * input_dim[1], 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), 32 * x.size(2))
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


# Define Critic network
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv1d(input_dim[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * input_dim[1], 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), 32 * x.size(2))
        x = self.fc2(x)
        return x


# Experience replay class
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# Process YOLOv5 output function
def process_yolov5_output(yolo_output, num_classes, threshold=0.5):
    batch_size, num_anchors, num_params = yolo_output.shape
    assert num_params == (4 + 1 + num_classes)

    small_scale_features = yolo_output[:, -1 * 20 * 20:].view(batch_size, 20 * 20, num_params)
    small_scale_class_probs = torch.softmax(small_scale_features[:, :, 5:], dim=2)
    small_scale_class_scores, small_scale_class_labels = torch.max(small_scale_class_probs, dim=2)

    small_scale_class_labels[small_scale_class_scores < threshold] = -1
    features = torch.cat(
        [small_scale_features[:, :, :4], small_scale_class_labels.unsqueeze(2), small_scale_class_scores.unsqueeze(2)],
        dim=2)

    return features.permute(0, 2, 1)


# Training function
def train(actor, critic, optimizer_actor, optimizer_critic, replay_buffer, num_episodes=1000, gamma=0.99,
          batch_size=64):
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state = process_yolov5_output(torch.randn(1, 128520, 15), 10).to(device)
        done = False
        total_reward = 0

        while not done:
            action_probs = actor(state)
            action = np.random.choice(len(action_probs[0]), p=action_probs.detach().cpu().numpy()[0])

            reward = random.uniform(-1, 1)
            next_state = process_yolov5_output(torch.randn(1, 128520, 15), 10).to(device)
            done = random.random() < 0.1

            total_reward += reward
            replay_buffer.add((state.clone(), action, reward, next_state.clone(), done))
            state = next_state

            if replay_buffer.size() > batch_size:
                experiences = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*experiences)

                states = torch.cat(states).to(device)
                next_states = torch.cat(next_states).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                with torch.no_grad():
                    target = rewards + (1 - dones) * gamma * critic(next_states)

                values = critic(states)
                critic_loss = nn.MSELoss()(values, target)
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                advantage = target - values.detach()
                actor_loss = -torch.log(action_probs[0][action]) * advantage.mean()
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")


# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


# Save model parameters
def save_model(actor, critic, actor_path='actor.pth', critic_path='critic.pth'):
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    print("Model parameters saved.")


# Load model parameters
def load_model(actor, critic, actor_path='actor.pth', critic_path='critic.pth'):
    try:
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        print("Model parameters loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
