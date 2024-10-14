def main():
    # 创建游戏环境实例
    env = GameEnvironment()

    # 创建DQN代理实例
    agent = DQNAgent(
        state_size=4,  # 状态向量的维度
        action_size=2,  # 动作空间的大小（例如：左移和右移）
        learning_rate=0.001,  # 学习率
        gamma=0.99,  # 折扣因子
        epsilon=1.0,  # 初始探索率
        buffer_size=10000,  # 经验回放缓冲区的大小
        batch_size=32  # 训练时的批量大小
    )

    # 创建训练管理器实例
    trainer = Training(agent, env, episodes=1000)

    # 开始训练过程
    trainer.train()


if __name__ == "__main__":
    main()