def reward_function(state, action, next_state):
    # 假设如果子弹和玩家的x坐标相同，则认为被击中
    player_pos = state[:2]
    bullet_pos = state[2:]
    next_player_pos = next_state[:2]
    next_bullet_pos = next_state[2:]

    reward = 0
    if player_pos[0] == bullet_pos[0]:
        reward = -1  # 被击中的惩罚
    elif next_player_pos[0] != next_bullet_pos[0]:
        reward = 1  # 躲避子弹的奖励
    else:
        reward = -0.1  # 未躲避子弹的小惩罚

    return reward