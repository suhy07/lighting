import os
import torch
from tqdm import tqdm

class Training:
    def __init__(self, agent, env, episodes, save_dir='models'):
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.save_dir = save_dir
        self.best_reward = float('-inf')
        
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def train(self):
        # 使用tqdm显示训练进度
        progress_bar = tqdm(range(self.episodes), desc='Training')
        episode_rewards = []
        
        for episode in progress_bar:
            state = self.env.reset()
            episode_reward = 0
            
            while not self.env.is_game_over():
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                self.agent.buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                # 更新智能体
                loss = self.agent.update()
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # 更新进度条信息
            progress_bar.set_postfix({
                'Reward': f'{episode_reward:.2f}',
                'Epsilon': f'{self.agent.epsilon:.2f}',
                'Loss': f'{loss:.4f}' if loss else 'N/A'
            })
            
            # 保存最佳模型
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model('best_model.pth')
            
            # 定期保存检查点
            if (episode + 1) % 100 == 0:
                self.save_model(f'checkpoint_episode_{episode+1}.pth')

    def save_model(self, filename):
        """保存模型到指定文件"""
        model_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.agent.model.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon,
            'best_reward': self.best_reward
        }, model_path)

    def load_model(self, filename):
        """从文件加载模型"""
        model_path = os.path.join(self.save_dir, filename)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.agent.model.load_state_dict(checkpoint['model_state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.agent.epsilon = checkpoint['epsilon']
            self.best_reward = checkpoint['best_reward']
            print(f'Successfully loaded model from {model_path}')
        else:
            print(f'No model found at {model_path}')