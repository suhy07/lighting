import os
import torch
import numpy as np
from thunder_drl import ThunderDRLTrainer
import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from DQN.ThunderGameEnv import ThunderGameEnv

def train_thunder_ai(episodes=1000, epsilon_start=0.1, epsilon_end=0.01,
                    epsilon_decay=0.995, save_path='./models', load_path='./models'):
    # 初始化环境
    env = ThunderGameEnv()
    
    # 获取状态和动作空间大小
    state_size = len(env.get_state())
    action_size = 8  # 8个方向的移动：上、右上、右、右下、下、左下、左、左上
    
    # 初始化DRL训练器
    trainer = ThunderDRLTrainer(
        env=env,
        state_size=state_size,
        action_size=action_size,
        actor_lr=5e-5,  # 降低学习率以提高稳定性
        critic_lr=5e-4,
        gamma=0.99,
        buffer_size=200000,  # 增加经验回放缓冲区大小
        batch_size=128,  # 增大批处理大小
        update_every=2  # 更频繁地更新网络
    )
    
    # 加载已保存的模型
    if os.path.exists(os.path.join(load_path, 'thunder_actor.pth')):
        trainer.load(load_path)
        print('已加载保存的模型参数')
    
    # 训练循环
    epsilon = epsilon_start
    best_reward = float('-inf')
    scores = []
    
    print('开始训练...')
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = trainer.act(state, epsilon)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 存储经验并学习
            trainer.step(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
        
        # 更新探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 记录得分
        scores.append(episode_reward)
        avg_score = np.mean(scores[-100:])  # 最近100轮的平均分数
        
        # 打印训练信息
        if episode % 10 == 0:
            print(f'Episode {episode}/{episodes}, Score: {episode_reward:.2f}, '
                  f'Avg Score: {avg_score:.2f}, Epsilon: {epsilon:.2f}')
        
        # 保存最佳模型
        if avg_score > best_reward:
            best_reward = avg_score
            trainer.save(save_path)
            print(f'保存新的最佳模型，平均得分: {best_reward:.2f}')
    
    print('训练完成！')
    return trainer

if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 开始训练
    trainer = train_thunder_ai()