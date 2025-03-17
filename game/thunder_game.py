import curses
import random
import time
from typing import List
from config import (ENEMY_SPAWN_DELAY, GAME_UPDATE_DELAY, BOSS_SCORE_THRESHOLD,
                   ENEMY_SCORE, BOSS_SCORE, CRYSTAL_SCORE, SHIELD_SCORE,
                   UPGRADE_SCORE, SPECIAL_SCORE)
from game_object import GameObject
from player import Player, Bullet
from enemy import Enemy, Boss, Crystal, Item

class ThunderGame:
    def __init__(self):
        self.screen = curses.initscr()
        curses.start_color()
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(True)
        self.screen.nodelay(1)
        
        self.max_y, self.max_x = self.screen.getmaxyx()
        self.player = Player(self.max_x // 2, self.max_y - 2)
        self.enemies: List[Enemy] = []
        self.boss: Boss = None
        self.crystals: List[Crystal] = []
        self.items: List[Item] = []
        self.score = 1490
        self.enemy_spawn_delay = 2.0
        self.last_enemy_spawn = 0
        self.game_over = False

    def spawn_enemy(self, current_time: float) -> None:
        if current_time - self.last_enemy_spawn >= self.enemy_spawn_delay:
            x = random.randint(0, self.max_x - 1)
            self.enemies.append(Enemy(x, 0))
            self.last_enemy_spawn = current_time

    def check_collisions(self) -> None:
        # Check bullet collisions with enemies
        for bullet in self.player.bullets[:]:
            for enemy in self.enemies[:]:
                if bullet.x == enemy.x and bullet.y == enemy.y:
                    try:
                        self.player.bullets.remove(bullet)
                        self.enemies.remove(enemy)
                        self.score += ENEMY_SCORE
                        # 生成水晶
                        self.crystals.extend(enemy.generate_crystals())
                        # 生成道具
                        item = enemy.generate_item()
                        if item:
                            self.items.append(item)
                    except ValueError:
                        pass
                    break

            # Check bullet collisions with boss
            if self.boss and self.boss.x <= bullet.x < self.boss.x + self.boss.width and \
               self.boss.y <= bullet.y < self.boss.y + self.boss.height:
                try:
                    self.player.bullets.remove(bullet)
                    self.boss.health -= 1
                    if self.boss.health <= 0:
                        # 生成大量水晶和道具
                        self.crystals.extend(self.boss.generate_crystals())
                        self.items.extend(self.boss.generate_items())
                        self.boss = None
                        self.score += BOSS_SCORE
                except ValueError:
                    pass

        # Check player collision with enemies
        for enemy in self.enemies:
            if self.player.x == enemy.x and self.player.y == enemy.y:
                self.game_over = True

        # Check player collision with boss
        if self.boss:
            boss_start_x = max(0, self.boss.x - (len(self.boss.symbol) // 2))
            if boss_start_x <= self.player.x < boss_start_x + self.boss.width and \
               self.boss.y <= self.player.y < self.boss.y + self.boss.height:
                self.game_over = True
        
        # Check player collision with boss bullets
        if self.boss:
            for bullet in self.boss.bullets:
                if bullet.x == self.player.x and bullet.y == self.player.y:
                    self.game_over = True

    def update(self) -> None:
        current_time = time.time()

        # Update player and bullets
        self.player.update(current_time)
        for bullet in self.player.bullets[:]:
            if bullet.update(self.max_x, self.max_y):
                self.player.bullets.remove(bullet)

        # Update enemies and crystals
        for enemy in self.enemies[:]:
            # 更新敌人子弹
            for bullet in enemy.bullets[:]:
                if bullet.update(self.max_x, self.max_y):
                    enemy.bullets.remove(bullet)
                elif bullet.x == self.player.x and bullet.y == self.player.y:
                    self.game_over = True

        # 更新水晶
        for crystal in self.crystals[:]:
            if crystal.update(current_time, self.player.x, self.player.y, self.max_x, self.max_y):
                self.crystals.remove(crystal)
                self.score += CRYSTAL_SCORE

        # 更新道具
        for item in self.items[:]:
            if item.update(current_time, self.player.x, self.player.y, self.max_x, self.max_y):
                self.items.remove(item)
                if item.item_type == 'shield':
                    self.score += SHIELD_SCORE
                elif item.item_type == 'upgrade':
                    self.score += UPGRADE_SCORE
                else:  # special
                    self.score += SPECIAL_SCORE

        for enemy in self.enemies[:]:
            if enemy.update(current_time, self.max_x, self.max_y):
                self.enemies.remove(enemy)

        # Spawn new enemies
        self.spawn_enemy(current_time)

        # Update boss
        if self.boss:
            self.boss.update(current_time, self.max_x, self.max_y, self.player.x, self.player.y)
            # 当Boss出现时降低敌人生成频率
            self.enemy_spawn_delay = ENEMY_SPAWN_DELAY * 2
        elif self.score >= BOSS_SCORE_THRESHOLD and not self.boss and \
             (self.score - BOSS_SCORE_THRESHOLD) % 1500 == 0:
            self.boss = Boss(self.max_x // 2, 2)
            self.enemy_spawn_delay = ENEMY_SPAWN_DELAY * 2

        # Check collisions
        self.check_collisions()

    def draw(self) -> None:
        self.screen.clear()

        # Draw player
        if 0 <= self.player.y < self.max_y and 0 <= self.player.x < self.max_x:
            try:
                self.screen.addch(self.player.y, self.player.x, self.player.symbol)
            except curses.error:
                pass

        # Draw player bullets
        for bullet in self.player.bullets:
            if 0 <= bullet.y < self.max_y and 0 <= bullet.x < self.max_x:
                try:
                    self.screen.addch(bullet.y, bullet.x, bullet.symbol)
                except curses.error:
                    pass

        # Draw enemies and their bullets
        for enemy in self.enemies:
            if 0 <= enemy.y < self.max_y and 0 <= enemy.x < self.max_x:
                try:
                    self.screen.addch(enemy.y, enemy.x, enemy.symbol)
                except curses.error:
                    pass
            for bullet in enemy.bullets:
                if 0 <= bullet.y < self.max_y and 0 <= bullet.x < self.max_x:
                    try:
                        self.screen.addch(bullet.y, bullet.x, bullet.symbol)
                    except curses.error:
                        pass

        # Draw crystals
        for crystal in self.crystals:
            if 0 <= crystal.y < self.max_y and 0 <= crystal.x < self.max_x:
                try:
                    self.screen.addch(crystal.y, crystal.x, crystal.symbol)
                except curses.error:
                    pass

        # Draw items
        for item in self.items:
            if 0 <= item.y < self.max_y and 0 <= item.x < self.max_x:
                try:
                    self.screen.addch(item.y, item.x, item.symbol)
                except curses.error:
                    pass

        # Draw boss and its bullets
        if self.boss:
            self.boss.draw(self.screen)
            for bullet in self.boss.bullets:
                if 0 <= bullet.y < self.max_y and 0 <= bullet.x < self.max_x:
                    try:
                        self.screen.addch(bullet.y, bullet.x, bullet.symbol)
                    except curses.error:
                        pass

        # Draw score
        score_text = f'Score: {self.score}'
        self.screen.addstr(0, 0, score_text)

        self.screen.refresh()

    def handle_input(self) -> bool:
        key = self.screen.getch()
        if key == ord('q'):
            return False
        # WASD movement with diagonal support
        dx = 0
        dy = 0
        if key == ord('a'):
            dx = -1
        elif key == ord('d'):
            dx = 1
        elif key == ord('w'):
            dy = -1
        elif key == ord('s'):
            dy = 1
        
        # Check for diagonal movement
        next_key = self.screen.getch()
        if next_key != -1:
            if next_key == ord('w') and dx != 0:
                dy = -1
            elif next_key == ord('s') and dx != 0:
                dy = 1
            elif next_key == ord('a') and dy != 0:
                dx = -1
            elif next_key == ord('d') and dy != 0:
                dx = 1
        
        if dx != 0 or dy != 0:
            self.player.move(dx, dy, self.max_x, self.max_y)
        return True

    def run(self) -> None:
        while not self.game_over:
            if not self.handle_input():
                break
            self.update()
            self.draw()
            time.sleep(0.05)

        # Game over screen
        self.screen.clear()
        game_over_text = 'Game Over! Final Score: {}'.format(self.score)
        self.screen.addstr(self.max_y // 2, (self.max_x - len(game_over_text)) // 2, game_over_text)
        self.screen.refresh()
        time.sleep(2)

    def cleanup(self) -> None:
        curses.nocbreak()
        self.screen.keypad(False)
        curses.echo()
        curses.endwin()

def main():
    game = ThunderGame()
    try:
        game.run()
    finally:
        game.cleanup()

if __name__ == '__main__':
    main()