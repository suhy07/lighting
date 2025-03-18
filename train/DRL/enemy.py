import time
import random
import math
import curses
from typing import List, Optional
from game_object import GameObject
from config import (ENEMY_SYMBOL, ENEMY_MOVE_DELAY, BOSS_SYMBOL, BOSS_MOVE_DELAY,
                   BOSS_MIN_HEALTH, BOSS_MAX_HEALTH, BULLET_SYMBOL, CRYSTAL_SYMBOL, CRYSTAL_ATTRACT_RANGE,
                   CRYSTAL_SPEED, CRYSTAL_COUNT, CRYSTAL_INITIAL_SPEED, ITEM_DROP_RATE,
                   ITEM_ATTRACT_RANGE, ITEM_SPEED, ITEM_INITIAL_SPEED, SHIELD_SYMBOL,
                   SHIELD_SCORE, UPGRADE_SYMBOL, UPGRADE_SCORE, SPECIAL_SYMBOL, SPECIAL_SCORE)

class EnemyBullet(GameObject):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, BULLET_SYMBOL)

    def update(self, max_x: int, max_y: int) -> bool:
        if self.y >= max_y - 1:
            return True
        self.move(0, 1, max_x, max_y)
        return False

class Crystal(GameObject):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, CRYSTAL_SYMBOL)
        self.last_move_time = 0
        # 初始化随机移动方向
        self.dx = random.choice([-1, 1]) * CRYSTAL_INITIAL_SPEED
        self.dy = random.random() + 0.5  # 确保向下移动
        self.initial_scatter = True

    def update(self, current_time: float, player_x: int, player_y: int, max_x: int, max_y: int) -> bool:
        # 计算与玩家的距离
        to_player_dx = player_x - self.x
        to_player_dy = player_y - self.y
        distance = math.sqrt(to_player_dx * to_player_dx + to_player_dy * to_player_dy)

        # 如果在吸附范围内，向玩家移动
        if distance <= CRYSTAL_ATTRACT_RANGE:
            if current_time - self.last_move_time >= CRYSTAL_SPEED:
                # 计算移动方向
                move_x = 1 if to_player_dx > 0 else -1 if to_player_dx < 0 else 0
                move_y = 1 if to_player_dy > 0 else -1 if to_player_dy < 0 else 0
                self.move(move_x, move_y, max_x, max_y)
                self.last_move_time = current_time
                self.initial_scatter = False
        # 否则执行散开和下落逻辑
        elif current_time - self.last_move_time >= CRYSTAL_SPEED:
            if self.initial_scatter:
                # 计算新位置，初始散开时使用较大的速度
                new_x = self.x + self.dx * 2
                new_y = self.y + self.dy
            else:
                # 正常下落时使用较小的速度
                new_x = self.x + self.dx * 0.5
                new_y = self.y + self.dy
            
            # 边界检查和反弹
            if new_x <= 0 or new_x >= max_x - 1:
                # 如果到达左右边界，返回True以移除水晶
                return True
            
            # 更新位置
            self.x = int(new_x)
            self.y = min(int(new_y), max_y - 1)
            self.last_move_time = current_time
            
            # 如果碰到底部，返回True以移除水晶
            if self.y >= max_y - 1:
                return True

        # 检查是否与玩家碰撞
        return self.x == player_x and self.y == player_y

class Item(GameObject):
    def __init__(self, x: int, y: int, item_type: str):
        self.item_type = item_type
        if item_type == 'shield':
            symbol = SHIELD_SYMBOL
            self.score = SHIELD_SCORE
        elif item_type == 'upgrade':
            symbol = UPGRADE_SYMBOL
            self.score = UPGRADE_SCORE
        else:  # special
            symbol = SPECIAL_SYMBOL
            self.score = SPECIAL_SCORE
        
        super().__init__(x, y, symbol)
        self.last_move_time = 0
        self.dx = random.choice([-1, 1]) * ITEM_INITIAL_SPEED
        self.dy = random.random() + 0.5
        self.initial_scatter = True

    def update(self, current_time: float, player_x: int, player_y: int, max_x: int, max_y: int) -> bool:
        to_player_dx = player_x - self.x
        to_player_dy = player_y - self.y
        distance = math.sqrt(to_player_dx * to_player_dx + to_player_dy * to_player_dy)

        if distance <= ITEM_ATTRACT_RANGE:
            if current_time - self.last_move_time >= ITEM_SPEED:
                move_x = 1 if to_player_dx > 0 else -1 if to_player_dx < 0 else 0
                move_y = 1 if to_player_dy > 0 else -1 if to_player_dy < 0 else 0
                self.move(move_x, move_y, max_x, max_y)
                self.last_move_time = current_time
                self.initial_scatter = False
        elif current_time - self.last_move_time >= ITEM_SPEED:
            if self.initial_scatter:
                new_x = self.x + self.dx * 2
                new_y = self.y + self.dy
            else:
                new_x = self.x + self.dx * 0.5
                new_y = self.y + self.dy
            
            if new_x <= 0 or new_x >= max_x - 1:
                return True
            
            self.x = int(new_x)
            self.y = min(int(new_y), max_y - 1)
            self.last_move_time = current_time
            
            if self.y >= max_y - 1:
                return True

        return self.x == player_x and self.y == player_y

class Enemy(GameObject):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, ENEMY_SYMBOL)
        self.move_delay = ENEMY_MOVE_DELAY
        self.last_move_time = 0
        self.last_shot_time = 0
        self.shoot_delay = 1.0
        self.bullets: List[EnemyBullet] = []
        self.direction = random.choice([-1, 0, 1])
        self.direction_change_time = 0
        self.direction_change_delay = 2.0
        self.max_x = 0
        self.max_y = 0

    def update(self, current_time: float, max_x: int, max_y: int) -> bool:
        # 更新最大边界值
        self.max_x = max_x
        self.max_y = max_y
        # 随机移动
        if current_time - self.direction_change_time >= self.direction_change_delay:
            self.direction = random.choice([-1, 0, 1])
            self.direction_change_time = current_time

        if current_time - self.last_move_time >= self.move_delay:
            new_x = self.x + self.direction
            if 0 <= new_x < max_x:
                self.move(self.direction, 1, max_x, max_y)
            else:
                self.move(0, 1, max_x, max_y)
            self.last_move_time = current_time

        # 发射子弹，降低发射频率
        if current_time - self.last_shot_time >= self.shoot_delay * 2:
            self.bullets.append(EnemyBullet(self.x, self.y + 1))
            self.last_shot_time = current_time

        # 更新子弹
        self.bullets = [bullet for bullet in self.bullets if not bullet.update(max_x, max_y)]

        # 当敌机到达屏幕底部时返回True，触发移除逻辑
        return self.y >= max_y - 1

    def generate_crystals(self) -> List[Crystal]:
        crystals = []
        for _ in range(CRYSTAL_COUNT):
            # 在敌机位置周围随机生成水晶
            crystal_x = max(0, min(self.x + random.randint(-2, 2), self.max_x - 1))
            crystal_y = max(0, min(self.y + random.randint(-1, 1), self.max_y - 1))
            crystals.append(Crystal(crystal_x, crystal_y))
        return crystals

    def generate_item(self) -> Optional[Item]:
        if random.random() < ITEM_DROP_RATE:
            item_type = random.choice(['shield', 'upgrade', 'special'])
            item_x = max(0, min(self.x + random.randint(-1, 1), self.max_x - 1))
            item_y = max(0, min(self.y + random.randint(-1, 1), self.max_y - 1))
            return Item(item_x, item_y, item_type)
        return None

class Boss(GameObject):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, BOSS_SYMBOL)
        self.health = random.randint(BOSS_MIN_HEALTH, BOSS_MAX_HEALTH)
        self.max_health = self.health  # 记录初始最大血量
        self.dx = random.choice([-1, 0, 1])
        self.dy = random.choice([-1, 0, 1])
        self.move_delay = BOSS_MOVE_DELAY
        self.last_move_time = 0
        self.last_shot_time = 0
        self.shoot_delay = 0.5
        self.bullets: List[EnemyBullet] = []
        self.max_x = 0
        self.max_y = 0
        # 初始化时设置宽度和高度为0，在update时根据屏幕大小动态调整
        self.width = 0
        self.height = 0
        self.direction_change_time = 0
        self.direction_change_delay = 2.0

    def update(self, current_time: float, max_x: int, max_y: int, player_x: int, player_y: int) -> None:
        self.max_x = max_x
        self.max_y = max_y
        # 动态调整Boss的大小为屏幕的三分之一
        self.width = max(3, max_x // 3)
        self.height = max(2, max_y // 3)
        
        # 根据血量调整移动模式
        health_percentage = self.health / self.max_health
        
        # 血量低于30%时，移动更加激进
        if health_percentage < 0.3:
            if random.random() < 0.2:  # 增加方向改变的概率
                self.dx = random.choice([-2, -1, 0, 1, 2])
                self.dy = random.choice([-1, 0, 1])
        # 正常移动模式
        elif current_time - self.direction_change_time >= self.direction_change_delay:
            self.dx = random.choice([-1, 0, 1])
            self.dy = random.choice([-1, 0, 1])
            self.direction_change_time = current_time
        
        if current_time - self.last_move_time >= self.move_delay:
            new_x = self.x + self.dx
            new_y = self.y + self.dy
            
            # 确保Boss不会完全离开屏幕
            min_x = 0
            max_x_pos = max_x - len(self.symbol)
            min_y = 0
            max_y_pos = max_y - self.height - 2  # 预留血条空间

            if new_x < min_x:
                new_x = min_x
                self.dx *= -1
            elif new_x > max_x_pos:
                new_x = max_x_pos
                self.dx *= -1

            if new_y < min_y:
                new_y = min_y
                self.dy *= -1
            elif new_y > max_y_pos:
                new_y = max_y_pos
                self.dy *= -1

            self.x = new_x
            self.y = new_y
            self.last_move_time = current_time

        # 发射子弹
        if current_time - self.last_shot_time >= self.shoot_delay:
            # 根据血量调整射击模式
            if health_percentage < 0.3:
                # 血量低时，发射更密集的子弹
                for x_offset in range(0, self.width, 2):
                    bullet_x = self.x + x_offset
                    if 0 <= bullet_x < max_x:
                        self.bullets.append(EnemyBullet(bullet_x, self.y + self.height))
                self.shoot_delay = 0.3  # 加快射击速度
            else:
                # 正常射击模式
                for x_offset in range(0, self.width, 3):
                    if random.random() < 0.4:  # 40%的概率从每个位置发射子弹
                        bullet_x = self.x + x_offset
                        if 0 <= bullet_x < max_x:
                            self.bullets.append(EnemyBullet(bullet_x, self.y + self.height))
                self.shoot_delay = 0.5

            self.last_shot_time = current_time

        # 更新子弹
        self.bullets = [bullet for bullet in self.bullets if not bullet.update(max_x, max_y)]

    def draw(self, screen) -> None:
        # 绘制Boss主体
        boss_str = self.symbol
        # 计算Boss的起始位置，确保居中显示
        start_x = max(0, min(self.x - (len(boss_str) // 2), self.max_x - len(boss_str)))
        start_y = max(0, min(self.y, self.max_y - self.height - 2))  # 预留血条和边界空间
        
        # 在Boss的范围内重复绘制符号
        for y in range(self.height):
            current_y = start_y + y
            if current_y >= self.max_y - 1:
                break
            for x in range(0, self.width, len(boss_str)):
                current_x = start_x + x
                if current_x >= self.max_x - len(boss_str):
                    break
                try:
                    screen.addstr(current_y, current_x, boss_str)
                except curses.error:
                    continue
        
        # 绘制血条
        health_bar_y = start_y + self.height
        if health_bar_y < self.max_y - 1:
            try:
                # 确保血条宽度不超出屏幕
                health_bar_width = min(self.width, self.max_x - start_x - 10)  # 预留百分比显示空间
                health_percentage = self.health / self.max_health
                filled_width = max(0, min(int(health_percentage * health_bar_width), health_bar_width))
                empty_width = health_bar_width - filled_width
                percentage_str = f"{int(health_percentage * 100)}%"
                health_bar = f"[{'=' * filled_width}{' ' * empty_width}] {percentage_str}"
                
                # 确保血条不会超出屏幕边界
                if start_x + len(health_bar) <= self.max_x:
                    screen.addstr(health_bar_y, start_x, health_bar)
            except curses.error:
                pass

    def generate_crystals(self) -> List[Crystal]:
        crystals = []
        for _ in range(CRYSTAL_COUNT * 2):  # Boss掉落双倍水晶
            crystal_x = max(0, min(self.x + random.randint(-2, 2), self.max_x - 1))
            crystal_y = max(0, min(self.y + random.randint(-1, 1), self.max_y - 1))
            crystals.append(Crystal(crystal_x, crystal_y))
        return crystals

    def generate_items(self) -> List[Item]:
        items = []
        # Boss必定掉落所有类型的道具
        for item_type in ['shield', 'upgrade', 'special']:
            item_x = max(0, min(self.x + random.randint(-1, 1), self.max_x - 1))
            item_y = max(0, min(self.y + random.randint(-1, 1), self.max_y - 1))
            items.append(Item(item_x, item_y, item_type))
        return items
