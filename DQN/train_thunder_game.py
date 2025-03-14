from DQNAgent import DQNAgent
from ThunderGameEnv import ThunderGameEnv
from Training import Training

# 初始化游戏环境
env = ThunderGameEnv()

# 设置训练参数
state_size = len(env.get_state())  # 状态空间大小
action_size = 3  # 动作空间大小：左移、右移、发射子弹
learning_rate = 0.001
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 初始探索率
buffer_size = 10000
batch_size = 64
episodes = 1000

# 初始化DQN智能体
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    learning_rate=learning_rate,
    gamma=gamma,
    epsilon=epsilon,
    buffer_size=buffer_size,
    batch_size=batch_size
)

# 创建训练器并开始训练
trainer = Training(agent, env, episodes)
print('开始训练...')
trainer.train()
print('训练完成！')