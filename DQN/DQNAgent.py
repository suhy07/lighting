import torch
import torch.nn as nn
import torch.optim as optim


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon, buffer_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        q_values = self.model(torch.FloatTensor(state))
        return np.argmax(q_values.detach().numpy())

    def update(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
