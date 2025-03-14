import numpy as np
import random

class ThunderGameEnv:
    def __init__(self):
        # 游戏区域大小
        self.width = 800
        self.height = 600
        
        # 玩家飞机属性
        self.player_width = 50
        self.player_height = 50
        self.player_speed = 20
        
        # 敌机属性
        self.enemy_width = 40
        self.enemy_height = 40
        self.enemy_speed = 5
        self.max_enemies = 3
        
        # 子弹属性
        self.bullet_width = 10
        self.bullet_height = 20
        self.bullet_speed = 15
        self.max_bullets = 5
        
        self.reset()
    
    def reset(self):
        # 初始化玩家位置
        self.player_pos = np.array([self.width // 2, self.height - self.player_height * 2])
        
        # 初始化敌机
        self.enemies = []
        for _ in range(self.max_enemies):
            self.spawn_enemy()
        
        # 初始化子弹
        self.bullets = []
        
        # 游戏状态
        self.score = 0
        self.game_over = False
        
        return self.get_state()
    
    def spawn_enemy(self):
        x = random.randint(0, self.width - self.enemy_width)
        y = random.randint(-self.enemy_height, 0)
        self.enemies.append(np.array([x, y]))
    
    def step(self, action):
        if self.game_over:
            return self.get_state(), 0, True
        
        # 处理动作
        # 0: 左移, 1: 右移, 2: 发射子弹, 3: 不动
        if action == 0:  # 左移
            self.player_pos[0] = max(0, self.player_pos[0] - self.player_speed)
        elif action == 1:  # 右移
            self.player_pos[0] = min(self.width - self.player_width, self.player_pos[0] + self.player_speed)
        elif action == 2:  # 发射子弹
            if len(self.bullets) < self.max_bullets:
                bullet_pos = np.array([self.player_pos[0] + self.player_width/2 - self.bullet_width/2,
                                      self.player_pos[1]])
                self.bullets.append(bullet_pos)
        
        # 更新子弹位置
        new_bullets = []
        for bullet in self.bullets:
            bullet[1] -= self.bullet_speed
            if bullet[1] >= 0:
                new_bullets.append(bullet)
        self.bullets = new_bullets
        
        # 更新敌机位置
        for enemy in self.enemies[::]:
            enemy[1] += self.enemy_speed
            if enemy[1] > self.height:
                if any(np.array_equal(enemy, e) for e in self.enemies):
                    self.enemies.remove(enemy)
                    self.spawn_enemy()
                    self.score -= 1  # 敌机逃脱扣分
        
        # 检测子弹与敌机的碰撞
        new_bullets = []
        new_enemies = []
        for bullet in self.bullets:
            bullet_hit = False
            for enemy in self.enemies:
                if self.check_collision(bullet, self.bullet_width, self.bullet_height,
                                      enemy, self.enemy_width, self.enemy_height):
                    bullet_hit = True
                    if enemy not in new_enemies:
                        self.spawn_enemy()
                        self.score += 2  # 击中敌机加分
                    break
            if not bullet_hit:
                new_bullets.append(bullet)
        
        for enemy in self.enemies:
            if not any(self.check_collision(bullet, self.bullet_width, self.bullet_height,
                                          enemy, self.enemy_width, self.enemy_height)
                      for bullet in self.bullets):
                new_enemies.append(enemy)
        
        self.bullets = new_bullets
        self.enemies = new_enemies
        
        # 检测玩家与敌机的碰撞
        for enemy in self.enemies:
            if self.check_collision(self.player_pos, self.player_width, self.player_height,
                                  enemy, self.enemy_width, self.enemy_height):
                self.game_over = True
                return self.get_state(), -10, True  # 游戏结束给予大的负奖励
        
        reward = 0.1  # 存活奖励
        if len(self.bullets) < self.max_bullets:
            reward += 0.05  # 鼓励发射子弹
        
        return self.get_state(), reward + self.score * 0.1, False
    
    def check_collision(self, pos1, width1, height1, pos2, width2, height2):
        return (pos1[0] < pos2[0] + width2 and
                pos1[0] + width1 > pos2[0] and
                pos1[1] < pos2[1] + height2 and
                pos1[1] + height1 > pos2[1])
    
    def get_state(self):
        # 状态包含：玩家位置，所有敌机位置，所有子弹位置
        state = [self.player_pos[0] / self.width, self.player_pos[1] / self.height]
        
        # 添加敌机位置
        enemy_state = np.zeros(self.max_enemies * 2)
        for i, enemy in enumerate(self.enemies):
            if i < self.max_enemies:
                enemy_state[i*2] = enemy[0] / self.width
                enemy_state[i*2+1] = enemy[1] / self.height
        state.extend(enemy_state)
        
        # 添加子弹位置
        bullet_state = np.zeros(self.max_bullets * 2)
        for i, bullet in enumerate(self.bullets):
            if i < self.max_bullets:
                bullet_state[i*2] = bullet[0] / self.width
                bullet_state[i*2+1] = bullet[1] / self.height
        state.extend(bullet_state)
        
        return np.array(state)
    
    def is_game_over(self):
        return self.game_over