from typing import List, Tuple

class GameObject:
    def __init__(self, x: int, y: int, symbol: str):
        self.x = x
        self.y = y
        self.symbol = symbol
        self.is_alive = True

    def move(self, dx: int, dy: int, max_x: int, max_y: int) -> None:
        new_x = self.x + dx
        new_y = self.y + dy
        if 0 <= new_x < max_x and 0 <= new_y < max_y:
            self.x = new_x
            self.y = new_y