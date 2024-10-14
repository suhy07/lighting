import numpy as np


class GameEnvironment:
    def __init__(self):
        self.player_pos = np.array([0.5, 0.5])
        self.bullet_pos = np.array([0.2, 0.5])
        self.game_over = False

    def reset(self):
        self.player_pos = np.array([0.5, 0.5])
        self.bullet_pos = np.array([0.2, 0.5])
        self.game_over = False
        return self.get_state()

    def step(self, action):
        if action == 0:
            self.player_pos[0] = max(0, self.player_pos[0] - 0.1)
        elif action == 1:
            self.player_pos[0] = min(1, self.player_pos[0] + 0.1)

        self.bullet_pos[0] += 0.1
        self.game_over = self.player_pos[0] == self.bullet_pos[0] + 0.05

        next_state = self.get_state()
        reward = -1 if self.game_over else 1
        return next_state, reward, self.game_over

    def get_state(self):
        return np.concatenate((self.player_pos, self.bullet_pos))

    def is_game_over(self):
        return self.game_over
