import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
from typing import Tuple, Dict, Any
from thunder_game import ThunderGame

class ThunderGameEnv:
    def __init__(self):
        self.game = None
        self.show_ui = False
        self.ui_game = None
        self.ui_process = None
        self.reset()

    def render(self, episode=None, epsilon=None, reward=None):
        # UI在单独的窗口中渲染，这里不需要做任何事
        # 格式化训练状态信息
        status = []
        if episode:
            status.append(f'Episode: {episode}')
            status.append(f'Epsilon: {epsilon:.3f}')
            status.append(f'Reward: {reward:.2f}')
            # 使用ANSI转义序列清除当前行并移动到行首
            sys.stdout.write('\033[2K\r' + ' | '.join(status))
            sys.stdout.flush()
    def reset(self) -> np.ndarray:
        """重置环境并返回初始状态"""
        if self.game:
            self.game.cleanup()
            # 等待一小段时间确保curses正确清理
            import time
            time.sleep(0.1)
        self.game = ThunderGame()
        # 确保游戏状态完全重置
        self.game.score = 0
        self.game.enemies.clear()
        self.game.crystals.clear()
        self.game.items.clear()
        self.game.boss = None
        self.game.game_over = False
        # 重置时关闭可能存在的旧UI进程
        if self.ui_process:
            self.ui_process.terminate()
            self.ui_process = None
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作并返回下一个状态、奖励、是否结束和额外信息"""
        # 检测按键
        if self.game.screen:
            key = self.game.screen.getch()
            if key == ord('s') and not self.show_ui:
                self.show_ui = True
                import subprocess
                import sys
                import json
                import tempfile
                
                # 创建临时文件来存储游戏状态
                state_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
                game_state = {
                    'player': {'x': self.game.player.x, 'y': self.game.player.y},
                    'enemies': [{'x': e.x, 'y': e.y} for e in self.game.enemies],
                    'boss': {'x': self.game.boss.x, 'y': self.game.boss.y} if self.game.boss else None,
                    'crystals': [{'x': c.x, 'y': c.y} for c in self.game.crystals],
                    'items': [{'x': i.x, 'y': i.y} for i in self.game.items],
                    'score': self.game.score
                }
                json.dump(game_state, state_file)
                state_file.close()
                
                # 在新窗口中启动UI游戏
                # 使用共享状态文件实时同步
                cmd = [sys.executable, 'ui_window.py', state_file.name]
                self.ui_process = subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))},
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                )
            elif key == ord('q') and self.show_ui:
                if self.ui_process:
                    self.ui_process.terminate()
                    self.ui_process = None
                self.show_ui = False

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
        
        # 游戏结束时关闭UI进程
        if self.game.game_over and self.ui_process:
            self.ui_process.terminate()
            self.ui_process = None
        return np.array(state, dtype=np.float32)

    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action index to movement direction
        Action space:
        0: stay
        1: up
        2: up-right
        3: right
        4: down-right
        5: down
        6: down-left
        7: left
        8: up-left
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
            reward += self.game.score * 0.02
        
        # 收集水晶和道具的奖励
        for crystal in self.game.crystals:
            if crystal.x == self.game.player.x and crystal.y == self.game.player.y:
                reward += 2.0
            else:
                # 接近水晶的奖励
                distance = abs(crystal.x - self.game.player.x) + abs(crystal.y - self.game.player.y)
                if distance < 5:
                    reward += 0.2 * (5 - distance)
        
        for item in self.game.items:
            if item.x == self.game.player.x and item.y == self.game.player.y:
                reward += 3.0
            else:
                # 接近道具的奖励
                distance = abs(item.x - self.game.player.x) + abs(item.y - self.game.player.y)
                if distance < 5:
                    reward += 0.3 * (5 - distance)
        
        # 躲避敌人的奖励
        for enemy in self.game.enemies:
            distance = abs(enemy.x - self.game.player.x) + abs(enemy.y - self.game.player.y)
            if distance < 3:
                reward -= 0.5 * (3 - distance)
        
        # Boss相关奖励
        if self.game.boss:
            if self.game.boss.health <= 0:
                reward += 15.0  # 击败Boss的额外奖励
            else:
                # 与Boss保持适当距离的奖励
                distance = abs(self.game.boss.x - self.game.player.x) + abs(self.game.boss.y - self.game.player.y)
                if 3 <= distance <= 6:
                    reward += 0.3
                elif distance < 2:
                    reward -= 1.0
        
        # 死亡惩罚
        if self.game.game_over:
            reward -= 15.0
        
        return reward

    def close(self):
        """Close environment"""
        if self.game:
            self.game.cleanup()
        if self.ui_game:
            self.ui_game.cleanup()