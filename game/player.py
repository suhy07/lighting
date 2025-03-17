import time
from typing import List
from game_object import GameObject
from config import PLAYER_SYMBOL, PLAYER_SHOOT_DELAY

class Bullet(GameObject):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, '|')

    def update(self, max_x: int, max_y: int) -> bool:
        if self.y <= 0:
            return True
        self.move(0, -1, max_x, max_y)
        return False

class Player(GameObject):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, PLAYER_SYMBOL)
        self.score = 0
        self.bullets: List[Bullet] = []
        self.last_shot_time = 0
        self.shoot_delay = PLAYER_SHOOT_DELAY

    def update(self, current_time: float) -> None:
        # Auto shooting
        if current_time - self.last_shot_time >= self.shoot_delay:
            self.bullets.append(Bullet(self.x, self.y - 1))
            self.last_shot_time = current_time