class Training:
    def __init__(self, agent, env, episodes):
        self.agent = agent
        self.env = env
        self.episodes = episodes

    def train(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            while not self.env.is_game_over():
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                self.agent.buffer.push(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            self.agent.update()
            if episode % 100 == 0:
                print(f'Episode {episode}, Epsilon: {self.agent.epsilon}')