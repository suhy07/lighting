import numpy as np
from typing import Tuple, Dict, Any
from game.thunder_game import ThunderGame
from game.player import Player
from game.enemy import Enemy, Boss

class ThunderGameEnv:
    def __init__(self):
        self.game = None
        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境并返回初始状态"""
        if self.game:
            self.game.cleanup()
        self.game = ThunderGame()
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作并返回下一个状态、奖励、是否结束和额外信息"""
        # 解析动作
        dx, dy = self._decode_action(action)
        
        # 移动玩家
        self.game.player.move(dx, dy, self.game.max_x, self.game.max_y)
        
        # 更新游戏状态
        self.game.update()
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查游戏是否结束
        done = self.game.game_over
        
        # 额外信息
        info = {
            'score': self.game.score
        }
        
        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """获取当前游戏状态的向量表示"""
        state = []
        
        # 玩家位置（归一化到[0,1]范围）
        state.extend([
            self.game.player.x / self.game.max_x,
            self.game.player.y / self.game.max_y
        ])
        
        # 最近的敌人位置和距离（最多3个）
        enemy_positions = []
        for enemy in sorted(self.game.enemies,
                          key=lambda e: abs(e.x - self.game.player.x) + abs(e.y - self.game.player.y))[:3]:
            enemy_positions.extend([
                enemy.x / self.game.max_x,
                enemy.y / self.game.max_y,
                (enemy.x - self.game.player.x) / self.game.max_x,
                (enemy.y - self.game.player.y) / self.game.max_y
            ])
        
        # 填充不足3个敌人的位置
        while len(enemy_positions) < 12:  # 3个敌人 * 4个特征
            enemy_positions.extend([0, 0, 0, 0])
        
        state.extend(enemy_positions)
        
        # Boss信息
        if self.game.boss:
            state.extend([
                self.game.boss.x / self.game.max_x,
                self.game.boss.y / self.game.max_y,
                self.game.boss.health / self.game.boss.max_health
            ])
        else:
            state.extend([0, 0, 0])
        
        # 最近的水晶和道具位置（最多2个）
        item_positions = []
        all_items = sorted(
            self.game.crystals + self.game.items,
            key=lambda i: abs(i.x - self.game.player.x) + abs(i.y - self.game.player.y)
        )[:2]
        
        for item in all_items:
            item_positions.extend([
                item.x / self.game.max_x,
                item.y / self.game.max_y
            ])
        
        # 填充不足2个物品的位置
        while len(item_positions) < 4:  # 2个物品 * 2个特征
            item_positions.extend([0, 0])
        
        state.extend(item_positions)
        
        return np.array(state, dtype=np.float32)

    def _decode_action(self, action: int) -> Tuple[int, int]:
        """将动作索引解码为移动方向
        动作空间：
        0: 不动
        1: 上
        2: 右上
        3: 右
        4: 右下
        5: 下
        6: 左下
        7: 左
        8: 左上
        """
        action_map = {
            0: (0, 0),
            1: (0, -1),
            2: (1, -1),
            3: (1, 0),
            4: (1, 1),
            5: (0, 1),
            6: (-1, 1),
            7: (-1, 0),
            8: (-1, -1)
        }
        return action_map[action]

    def _calculate_reward(self) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 基础生存奖励
        reward += 0.1
        
        # 击败敌人的奖励
        if self.game.score > 0:
            reward += self.game.score * 0.01
        
        # 收集水晶和道具的奖励
        for crystal in self.game.crystals:
            if crystal.x == self.game.player.x and crystal.y == self.game.player.y:
                reward += 1.0
        
        for item in self.game.items:
            if item.x == self.game.player.x and item.y == self.game.player.y:
                reward += 2.0
        
        # 击败Boss的额外奖励
        if self.game.boss and self.game.boss.health <= 0:
            reward += 10.0
        
        # 死亡惩罚
        if self.game.game_over:
            reward -= 10.0
        
        return reward

    def close(self):
        """关闭环境"""
        if self.game:
            self.game.cleanup()