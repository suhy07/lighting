import numpy as np


# 假设状态空间是游戏角色和子弹的位置
# 状态向量示例：[player_x, player_y, bullet_x, bullet_y]
def get_state(player_pos, bullet_pos):
    return np.array([player_pos[0], player_pos[1], bullet_pos[0], bullet_pos[1]])


def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, len(Q[state]))
    else:
        # 返回具有最高Q值的动作
        return np.argmax(Q[state])
